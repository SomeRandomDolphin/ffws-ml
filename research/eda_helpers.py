"""helper utilities shared by research notebooks."""

from pathlib import Path

def _project_root() -> Path:
    """Resolve the repository root based on this helper location."""
    return Path(__file__).resolve().parents[1]

def _normalize_output_path(path: str | Path) -> Path:
    """Map bare filenames to the standard figures directory."""
    output = Path(path)
    if output.suffix == "":
        output = output.with_suffix(".png")

    if output.parent == Path("."):
        output = _project_root() / "reports" / "figures" / output.name

    output.parent.mkdir(parents=True, exist_ok=True)
    return output

def save_fig(fig, path, dpi: int = 150) -> None:
    """Save a matplotlib figure and print its final path."""
    output_path = _normalize_output_path(path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Disimpan: {output_path}")
