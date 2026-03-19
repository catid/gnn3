from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm
import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "reports" / "papers"
PDF_DIR = PAPERS_DIR / "pdf"
RAW_DIR = PAPERS_DIR / "md_raw"
CLEAN_DIR = PAPERS_DIR / "md_clean"
NOTES_DIR = PAPERS_DIR / "notes"
INDEX_PATH = PAPERS_DIR / "index.md"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class PaperSpec:
    slug: str
    title: str
    source_url: str
    pdf_url: str
    notes: tuple[str, str, str]


PAPERS: tuple[PaperSpec, ...] = (
    PaperSpec(
        slug="attention_residuals",
        title="Attention Residuals",
        source_url="https://arxiv.org/abs/2603.15031",
        pdf_url="https://arxiv.org/pdf/2603.15031.pdf",
        notes=(
            "Replaces fixed residual addition with learned depth-wise attention reads over earlier layers.",
            "Useful as an analogy for reading prior outer-refinement states or selective memory hubs across rounds.",
            "Most of the gain may be depth-mixing-specific rather than a direct fit for packet routing graphs.",
        ),
    ),
    PaperSpec(
        slug="mamba3",
        title="Mamba-3: Improved Sequence Modeling using State Space Principles",
        source_url="https://arxiv.org/abs/2603.15569",
        pdf_url="https://arxiv.org/pdf/2603.15569.pdf",
        notes=(
            "Pushes Mamba forward with improved discretization, complex-valued dynamics, and MIMO updates.",
            "Directly motivates a reusable transition operator, bidirectional scans, and multi-input fusion in the packet model.",
            "The full paper is sequence-centric and hardware-kernel-dependent, so a small pure-PyTorch adaptation is safer initially.",
        ),
    ),
    PaperSpec(
        slug="recursive_stem_model",
        title="Form Follows Function: Recursive Stem Model",
        source_url="https://arxiv.org/abs/2603.15641",
        pdf_url="https://arxiv.org/pdf/2603.15641.pdf",
        notes=(
            "Changes the recursive training contract: detached warm-up, terminal loss, independent H/L growth, and settling diagnostics.",
            "This is the most directly applicable source for outer refinement loops, curricula, and reliability signals.",
            "Benefits depend on having a transition operator that already makes local progress; otherwise terminal-only loss can stall.",
        ),
    ),
    PaperSpec(
        slug="jumping_knowledge_networks",
        title="Jumping Knowledge Networks",
        source_url="https://proceedings.mlr.press/v80/xu18c.html",
        pdf_url="https://proceedings.mlr.press/v80/xu18c/xu18c.pdf",
        notes=(
            "Introduces adaptive aggregation over intermediate graph-network depths instead of only the final layer.",
            "A clean precedent for outer-round-history reads and selective depth aggregation in routing refinement.",
            "JK helps preserve information but does not itself solve communication bottlenecks or dynamic routing.",
        ),
    ),
    PaperSpec(
        slug="graph_mamba_sequence_modeling",
        title="Graph-Mamba: Long-Range Graph Sequence Modeling with Selective State Spaces",
        source_url="https://arxiv.org/abs/2402.00789",
        pdf_url="https://arxiv.org/pdf/2402.00789.pdf",
        notes=(
            "Adapts selective state spaces to ordered graph sequences to improve long-range modeling.",
            "Relevant for graph ordering choices and for deciding how much sequential bias to inject before local mixing.",
            "Sequence-first graph reductions can lose locality unless the ordering is strongly aligned with the routing task.",
        ),
    ),
    PaperSpec(
        slug="graph_mamba",
        title="Graph Mamba: Towards Learning on Graphs with State Space Models",
        source_url="https://arxiv.org/abs/2402.08678",
        pdf_url="https://arxiv.org/pdf/2402.08678.pdf",
        notes=(
            "Frames graph-selective SSM design around neighborhood tokenization, ordering, local encoding, and bidirectional scans.",
            "This is the most directly relevant graph-SSM paper for the baseline packet backbone design.",
            "Reported gains rely on careful ordering and local encoders; naive graph-to-sequence conversions underperform.",
        ),
    ),
    PaperSpec(
        slug="denseformer",
        title="DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging",
        source_url="https://arxiv.org/abs/2402.02622",
        pdf_url="https://arxiv.org/pdf/2402.02622.pdf",
        notes=(
            "Uses learned depth-weighted aggregation to improve information flow across layers.",
            "Useful as a low-risk reference for aggregating outer-round states without fully general cross-depth attention.",
            "Depth averaging can blur state roles if used everywhere instead of at carefully chosen read points.",
        ),
    ),
    PaperSpec(
        slug="deepcrossattention",
        title="DeepCrossAttention: Cross-Layer Attention for Improved Deep Model Training",
        source_url="https://arxiv.org/abs/2502.06785",
        pdf_url="https://arxiv.org/pdf/2502.06785.pdf",
        notes=(
            "Applies explicit cross-layer attention over previous hidden states rather than fixed residual accumulation.",
            "Relevant for exploration around selective reads over prior outer rounds and role-conditioned depth mixing.",
            "Cross-layer retrieval adds cost and can become a crutch if the local transition dynamics are weak.",
        ),
    ),
    PaperSpec(
        slug="muddformer",
        title="MUDDFormer: Breaking Transformer Depth Limits with Multi-Scale Dynamic Depth Fusion",
        source_url="https://arxiv.org/abs/2502.12170",
        pdf_url="https://arxiv.org/pdf/2502.12170.pdf",
        notes=(
            "Targets deep-model optimization with multi-scale dynamic fusion across depth.",
            "Provides design ideas for mixing shallow and deep routing states when outer refinement depth grows at test time.",
            "The method is optimized for deep transformers, so only the depth-fusion concepts should transfer here.",
        ),
    ),
)


def download_pdf(spec: PaperSpec, *, force: bool) -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{spec.slug}.pdf"
    if path.exists() and not force:
        return path
    response = requests.get(spec.pdf_url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def extract_markdown(pdf_path: Path) -> str:
    return str(pymupdf4llm.to_markdown(pdf_path))


def clean_markdown(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    lines = [line.rstrip() for line in raw.splitlines()]
    counts = Counter(line.strip() for line in lines if line.strip())
    repeated_short = {
        line
        for line, count in counts.items()
        if count >= 3 and len(line) < 80 and not line.startswith("#")
    }
    filtered = [
        line
        for line in lines
        if line.strip() not in repeated_short and not re.fullmatch(r"\d+", line.strip())
    ]

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(part.strip() for part in buffer)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paragraphs.append(text)
        buffer.clear()

    block_prefixes = ("#", "-", "*", ">")
    for line in filtered:
        stripped = line.strip()
        if not stripped:
            flush()
            paragraphs.append("")
            continue
        if stripped.startswith(block_prefixes):
            flush()
            paragraphs.append(stripped)
            continue
        if re.match(r"^\d+\.", stripped):
            flush()
            paragraphs.append(stripped)
            continue
        if stripped.startswith("|") or stripped.endswith("|"):
            flush()
            paragraphs.append(stripped)
            continue
        if len(stripped) < 90 and stripped.isupper():
            flush()
            paragraphs.append(f"## {stripped.title()}")
            continue
        buffer.append(stripped)

    flush()

    cleaned = "\n".join(paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned + "\n"


def write_notes(spec: PaperSpec) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_path = NOTES_DIR / f"{spec.slug}.md"
    novel, applicable, risky = spec.notes
    note_path.write_text(
        "\n".join(
            [
                f"# {spec.title}",
                "",
                f"- Source: {spec.source_url}",
                "- What is actually novel:",
                f"  {novel}",
                "- What is directly applicable here:",
                f"  {applicable}",
                "- What is risky or likely hype:",
                f"  {risky}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_index(specs: list[PaperSpec]) -> None:
    lines = [
        "# Paper Index",
        "",
        "| slug | title | source | pdf | cleaned markdown | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.slug} | {spec.title} | [source]({spec.source_url}) | [pdf](pdf/{spec.slug}.pdf) | "
            f"[clean](md_clean/{spec.slug}.md) | [notes](notes/{spec.slug}.md) |"
        )
    INDEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_paper(spec: PaperSpec, *, force: bool) -> None:
    pdf_path = download_pdf(spec, force=force)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{spec.slug}.md"
    clean_path = CLEAN_DIR / f"{spec.slug}.md"

    raw_markdown = extract_markdown(pdf_path)
    raw_path.write_text(raw_markdown, encoding="utf-8")
    clean_path.write_text(clean_markdown(raw_markdown), encoding="utf-8")
    write_notes(spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract all papers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in PAPERS:
        process_paper(spec, force=args.force)
    write_index(list(PAPERS))


if __name__ == "__main__":
    main()
