"""Render the system architecture diagram to reports/figures/architecture.png."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from heartrisk.data import ROOT  # noqa: E402

LANES = [  # (title, y, colour)
    ("Data & Training", 7.6, "#dfe6fd"),
    ("CI/CD (GitHub Actions)", 5.2, "#fdeedc"),
    ("Serving (Kubernetes)", 2.8, "#dcf5ea"),
    ("Monitoring", 0.4, "#f6e1f3"),
]
BOXES = {
    "uci": (0.3, 7.9, "UCI repository\nprocessed.cleveland.data"),
    "dl": (3.0, 7.9, "download_data.py\nclean + validate"),
    "eda": (5.7, 7.9, "eda.py\nfigures + summary"),
    "train": (8.4, 7.9, "heartrisk.train\nGridSearchCV x 3 models"),
    "mlflow": (11.1, 7.9, "MLflow tracking\nparams/metrics/plots"),
    "model": (13.8, 7.9, "model.joblib\nfull sklearn Pipeline"),
    "push": (0.3, 5.5, "git push / PR"),
    "lint": (3.0, 5.5, "Lint\nruff"),
    "test": (5.7, 5.5, "Unit tests\npytest + coverage"),
    "ci_train": (8.4, 5.5, "Train + quality\ngate (AUC>=0.85)"),
    "build": (11.1, 5.5, "Docker build\n+ smoke test"),
    "art": (13.8, 5.5, "Artifacts\nmodel, mlruns, image"),
    "client": (0.3, 3.1, "Client\ncurl / Swagger"),
    "lb": (3.0, 3.1, "Service\nLoadBalancer :8080"),
    "pods": (5.7, 3.1, "Deployment (2-5 pods)\nFastAPI /predict"),
    "hpa": (8.4, 3.1, "HPA\nCPU 70%"),
    "logs": (11.1, 3.1, "JSON request logs\nstdout / kubectl logs"),
    "prom": (5.7, 0.7, "Prometheus\nscrapes /metrics"),
    "graf": (8.4, 0.7, "Grafana dashboard\nlatency, errors, drift"),
}
EDGES = [
    ("uci", "dl"), ("dl", "eda"), ("eda", "train"), ("train", "mlflow"), ("mlflow", "model"),
    ("push", "lint"), ("lint", "test"), ("test", "ci_train"), ("ci_train", "build"),
    ("build", "art"), ("client", "lb"), ("lb", "pods"), ("pods", "hpa"), ("hpa", "logs"),
    ("pods", "prom"), ("prom", "graf"), ("model", "art"), ("art", "pods"),
]  # fmt: skip
W, H = 2.3, 1.0


def centre(key):
    x, y, _ = BOXES[key]
    return x + W / 2, y + H / 2


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    for title, y, colour in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y), 16.2, 1.9, boxstyle="round,pad=0.02",
                                    fc=colour, ec="none", alpha=0.7))  # fmt: skip
        ax.text(0.2, y + 1.75, title, fontsize=11, fontweight="bold", va="top", color="#2d3436")
    for x, y, label in BOXES.values():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05",
                                    fc="white", ec="#2d3436", lw=1.3))  # fmt: skip
        ax.text(x + W / 2, y + H / 2, label, ha="center", va="center", fontsize=9)
    for a, b in EDGES:
        (x1, y1), (x2, y2) = centre(a), centre(b)
        if y1 == y2:
            start, end = (x1 + W / 2 + 0.05, y1), (x2 - W / 2 - 0.05, y2)
        else:
            sign = -1 if y2 < y1 else 1
            start, end = (x1, y1 + sign * (H / 2 + 0.05)), (x2, y2 - sign * (H / 2 + 0.05))
        rad = "arc3,rad=0.0" if y1 == y2 else "arc3,rad=0.15"
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color="#636e72", lw=1.4, connectionstyle=rad))  # fmt: skip
    ax.set_title("Heart Disease Risk - MLOps architecture", fontsize=15, fontweight="bold")
    out = ROOT / "reports" / "figures" / "architecture.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
