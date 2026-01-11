from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from Bio import AlignIO, Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def shell_quote(s: str) -> str:
    """Single-quote for safe use inside a bash -lc string."""
    return "'" + s.replace("'", r"'\''") + "'"


def win_to_wsl_path(p: Path) -> str:
    """
    Convert Windows path like C:\\dir\\file.faa to /mnt/c/dir/file.faa.
    If already POSIX, return unchanged.
    """
    s = str(p)
    if s.startswith("/"):
        return s
    s = s.replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:]
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return s


def ensure_unique_ids(records: List[SeqRecord]) -> List[SeqRecord]:
    seen = {}
    out = []
    for r in records:
        base = r.id.strip().replace(" ", "_")
        if not base:
            base = "seq"
        if base not in seen:
            seen[base] = 0
            new_id = base
        else:
            seen[base] += 1
            new_id = f"{base}__{seen[base]}"
        out.append(SeqRecord(Seq(str(r.seq)), id=new_id, description=""))
    return out


def run_mafft_native(in_faa: Path, out_faa: Path, extra_args: List[str] | None = None) -> None:
    exe = shutil.which("mafft")
    if not exe:
        raise FileNotFoundError("mafft executable not found on PATH (native).")
    out_faa.parent.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "--auto"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(in_faa))

    log_path = out_faa.with_suffix(out_faa.suffix + ".mafft.log")
    with open(out_faa, "w", newline="\n") as out_h, open(log_path, "w", newline="\n") as log_h:
        proc = subprocess.run(cmd, stdout=out_h, stderr=log_h, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed (exit {proc.returncode}). See log: {log_path}")

def run_mafft_wsl(in_faa: Path, out_faa: Path, extra_args: List[str] | None = None) -> None:
    """
    Robust WSL MAFFT runner:
    1) Copy input FASTA to /tmp inside Ubuntu (avoids /mnt/c quoting issues).
    2) Run mafft in /tmp.
    3) Copy output back to Windows path.
    """
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("wsl"):
        raise FileNotFoundError("wsl.exe not found. Install WSL or use native MAFFT.")

    out_faa.parent.mkdir(parents=True, exist_ok=True)

    # Windows -> WSL path converter
    def win_to_wsl_path(p: Path) -> str:
        s = str(p).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            rest = s[2:]
            if rest.startswith("/"):
                rest = rest[1:]
            return f"/mnt/{drive}/{rest}"
        return s

    # Use /tmp paths (no spaces)
    tmp_in = "/tmp/cssv_in.faa"
    tmp_out = "/tmp/cssv_out.faa"

    wsl_in = win_to_wsl_path(in_faa)
    wsl_out = win_to_wsl_path(out_faa)

    args = ["--auto"]
    if extra_args:
        args.extend(extra_args)

    # Build a single bash script so failure is visible
    bash_lines = [
        "set -e",
        f"cp {wsl_in!r} {tmp_in!r}",
        f"/usr/bin/mafft {' '.join(args)} {tmp_in!r} > {tmp_out!r}",
        f"cp {tmp_out!r} {wsl_out!r}",
    ]
    bash_cmd = "; ".join(bash_lines)

    log_path = out_faa.with_suffix(out_faa.suffix + ".wsl_mafft.log")
    with open(log_path, "w", newline="\n") as log_h:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "bash", "-lc", bash_cmd],
            stdout=subprocess.DEVNULL,
            stderr=log_h,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"WSL MAFFT failed (exit {proc.returncode}). See log: {log_path}")

def read_alignment(path: Path):
    return AlignIO.read(str(path), "fasta")


def trim_alignment_gappy_cols(aln, gap_cutoff: float = 0.5):
    n = len(aln)
    if n == 0:
        return aln
    L = aln.get_alignment_length()
    keep = []
    for i in range(L):
        gaps = 0
        for rec in aln:
            c = rec.seq[i]
            if c in {"-", "."}:
                gaps += 1
        if (gaps / n) <= gap_cutoff:
            keep.append(i)

    trimmed = []
    for rec in aln:
        seq = "".join(rec.seq[i] for i in keep)
        trimmed.append(SeqRecord(Seq(seq), id=rec.id, description=""))

    from Bio.Align import MultipleSeqAlignment
    return MultipleSeqAlignment(trimmed)


def pairwise_identity(seq_a: str, seq_b: str) -> float:
    matches = 0
    compared = 0
    for a, b in zip(seq_a, seq_b):
        if a in "-." or b in "-.":
            continue
        if a.upper() in {"X", "?"} or b.upper() in {"X", "?"}:
            continue
        compared += 1
        if a == b:
            matches += 1
    return (matches / compared) if compared else float("nan")


def identity_distance_matrix(aln) -> pd.DataFrame:
    names = [rec.id for rec in aln]
    n = len(names)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pid = pairwise_identity(str(aln[i].seq), str(aln[j].seq))
            d = 1.0 - pid if np.isfinite(pid) else float("nan")
            dist[i, j] = d
            dist[j, i] = d
    return pd.DataFrame(dist, index=names, columns=names)


def nj_tree_from_dist(dist_df: pd.DataFrame):
    names = dist_df.index.tolist()
    matrix = []
    for i in range(len(names)):
        row = []
        for j in range(i + 1):
            row.append(float(dist_df.iat[i, j]))
        matrix.append(row)

    dm = DistanceMatrix(names, matrix)
    tree = DistanceTreeConstructor().nj(dm)
    tree.ladderize()
    return tree


def plot_tree_pdf(tree, out_pdf: Path, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt

    n = len(tree.get_terminals())
    fig_h = max(4.0, 0.22 * n)
    fig_w = 10.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=ax, do_show=False)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)


def plot_heatmap(dist_df: pd.DataFrame, out_pdf: Path, dpi: int = 300, max_labels: int = 80) -> None:
    import matplotlib.pyplot as plt

    arr = dist_df.values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(arr, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n = dist_df.shape[0]
    if n <= max_labels:
        ax.set_xticks(range(n))
        ax.set_xticklabels(dist_df.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(dist_df.index, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("taxa (labels hidden; too many)")
        ax.set_ylabel("taxa (labels hidden; too many)")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Protein ORF MSA (MAFFT) + trim + identity distance + NJ tree (newick/pdf)"
    )
    ap.add_argument("--faa", required=True, help="Protein FASTA (e.g., longest_orfs.faa)")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument(
        "--aligner",
        default="auto",
        choices=["auto", "mafft", "wsl-mafft", "none"],
        help="MSA method. auto: native mafft then WSL. none: assume input already aligned.",
    )
    ap.add_argument("--gap_cutoff", type=float, default=0.5, help="Drop columns with gap fraction > cutoff")
    ap.add_argument("--max_heatmap_labels", type=int, default=80)
    args = ap.parse_args()

    faa = Path(args.faa)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(str(faa), "fasta"))
    if not records:
        raise FileNotFoundError(f"No sequences found in: {faa}")

    records = ensure_unique_ids(records)
    in_fixed = out_dir / "input.unique_ids.faa"
    SeqIO.write(records, str(in_fixed), "fasta")

    aln_path = out_dir / "alignment.mafft.faa"
    if args.aligner == "none":
        aln_path = in_fixed
    else:
        if args.aligner == "auto":
            if shutil.which("mafft"):
                run_mafft_native(in_fixed, aln_path)
            else:
                run_mafft_wsl(in_fixed, aln_path)
        elif args.aligner == "mafft":
            run_mafft_native(in_fixed, aln_path)
        elif args.aligner == "wsl-mafft":
            run_mafft_wsl(in_fixed, aln_path)

    aln = read_alignment(aln_path)

    trimmed = trim_alignment_gappy_cols(aln, gap_cutoff=args.gap_cutoff)
    trimmed_path = out_dir / f"alignment.trim_gap{args.gap_cutoff:.2f}.faa"
    AlignIO.write(trimmed, str(trimmed_path), "fasta")

    dist_df = identity_distance_matrix(trimmed)
    dist_csv = out_dir / "pairwise_identity_distance.csv"
    dist_df.to_csv(dist_csv)

    tree = nj_tree_from_dist(dist_df)
    tree_nwk = out_dir / "nj_tree.newick"
    Phylo.write(tree, str(tree_nwk), "newick")

    plot_tree_pdf(tree, out_dir / "nj_tree.pdf")
    plot_heatmap(dist_df, out_dir / "pairwise_distance_heatmap.pdf", max_labels=args.max_heatmap_labels)

    print("Done.")
    print(f"- unique ids: {in_fixed}")
    print(f"- alignment:  {aln_path}")
    print(f"- trimmed:    {trimmed_path}")
    print(f"- dist csv:   {dist_csv}")
    print(f"- tree:       {tree_nwk}")
    print(f"- tree pdf:   {out_dir / 'nj_tree.pdf'}")


if __name__ == "__main__":
    main()
