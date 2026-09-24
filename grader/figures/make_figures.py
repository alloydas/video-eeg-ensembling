"""Model diagrams for the paper: video grader (X3D-M, dual head), EEG detector (TCN), and the
EEG-Gated Racine Grader (EGRG) that joins them.

Writes fig_video_grader, fig_eeg_detector and fig_egrg_system as .pdf (vector, for LaTeX),
.svg and .png into this directory. Layer counts and tensor shapes were read from the code:
pytorchvideo x3d_m (stage depths 3/5/11/7; widths 24/48/96/192; head 192->432->pool->2048) and
train_pooled_eeg.TCN (5 residual blocks, 64 ch, k=7, dilations 1..16, receptive field 373 samples).

Usage: python grader/figures/make_figures.py      (matplotlib only; any cwd, no EEG_ROOT needed:
       the figures are always written next to this script, into grader/figures/)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
VID, EEG, GATE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1d2027", "#6b7079", "#d9dce2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "svg.fonttype": "none",
                     "pdf.fonttype": 42})


def box(ax, x, y, w, h, title, body="", color=INK, fill="#ffffff", lw=1.1, tsize=7.4, bsize=6.1,
        bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=lw, edgecolor=color, facecolor=fill))
    if body:
        ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=tsize,
                fontweight="bold" if bold else "normal", color=INK)
        ax.text(x + w / 2, y + h - 0.034, body, ha="center", va="top", fontsize=bsize,
                color=INK, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize,
                fontweight="bold" if bold else "normal", color=INK)


def arrow(ax, x0, y0, x1, y1, color=INK, label=None, lpos=0.5, dy=0.006, lsize=6.0, style="-|>",
          rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7,
                                 linewidth=0.9, color=color, shrinkA=0, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(x0 + (x1 - x0) * lpos, y0 + (y1 - y0) * lpos + dy, label, ha="center",
                va="bottom", fontsize=lsize, color=MUTED)


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_video():
    fig = plt.figure(figsize=(7.2, 2.3)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.3 / 7.2); ax.axis("off")
    ax.text(0.005, 0.305, "Video grader: X3D-M with shared 3- and 5-class heads (2.99 M parameters)",
            fontsize=8.4, fontweight="bold", color=INK, va="top")
    y, h = 0.135, 0.105
    blocks = [
        ("Input", "16 frames over\nthe whole clip\n3×16×224²", 0.005, 0.098, VID),
        ("Stem", "(2+1)D conv\nstride 2\n24×16×112²", 0.123, 0.085, INK),
        ("res2 ×3", "24×16×56²", 0.226, 0.076, INK),
        ("res3 ×5", "48×16×28²", 0.318, 0.076, INK),
        ("res4 ×11", "96×16×14²", 0.410, 0.076, INK),
        ("res5 ×7", "192×16×7²", 0.502, 0.076, INK),
        ("Pooled head", "conv → 432\npool 16×7×7\nconv → 2048", 0.594, 0.108, INK),
    ]
    for t, b, x, w, c in blocks:
        box(ax, x, y, w, h, t, b, color=c)
    for (_, _, x0, w0, _), (_, _, x1, _, _) in zip(blocks[:-1], blocks[1:]):
        arrow(ax, x0 + w0, y + h / 2, x1, y + h / 2)
    hx, hw, hh = 0.80, 0.195, 0.085
    box(ax, hx, 0.205, hw, hh, "3-class head", "Linear 2048 → 3\nnon-seizure, mild, severe", color=VID)
    box(ax, hx, 0.085, hw, hh, "5-class head", "Linear 2048 → 5\nnon-seizure, S2, S3, S4, S5", color=VID)
    arrow(ax, 0.702, y + h / 2, hx, 0.2475)
    arrow(ax, 0.702, y + h / 2, hx, 0.1275)
    ax.text(0.745, y + h / 2, "f ∈ ℝ²⁰⁴⁸", ha="center", va="center", fontsize=6.0, color=MUTED,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.text(0.005, 0.062, "Each res stage stacks X3D blocks (1×1×1 expand → 3×3×3 depthwise → squeeze-excite → "
            "1×1×1 project, residual); time stays at 16 frames.\nLoss: class-weighted CE on each head, summed. "
            "Fix: the Kinetics head's Softmax is removed, so both losses see logits.", fontsize=6.0,
            color=MUTED, va="top", linespacing=1.4)
    save(fig, "fig_video_grader")


def fig_eeg():
    fig = plt.figure(figsize=(7.2, 2.3)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.3 / 7.2); ax.axis("off")
    ax.text(0.005, 0.305, "EEG detector: TCN, seizure vs non-seizure, fixed last epoch (259.4 k parameters)",
            fontsize=8.4, fontweight="bold", color=INK, va="top")
    y, h = 0.135, 0.105
    box(ax, 0.005, y, 0.1, h, "EEG clip", "1 channel\n1 kHz\n59–105 s", color=EEG)
    box(ax, 0.12, y, 0.135, h, "Windows", "resample to 125 Hz\n6 s windows, 3 s hop\nz-score each window")
    arrow(ax, 0.105, y + h / 2, 0.12, y + h / 2)
    tx, tw = 0.27, 0.36
    ax.add_patch(FancyBboxPatch((tx, y - 0.012), tw, h + 0.05, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=1.1, edgecolor=EEG, facecolor="#fff7f2"))
    ax.text(tx + tw / 2, y + h + 0.031, "TCN, same weights for every window", ha="center", va="top",
            fontsize=7.4, fontweight="bold")
    bw, gap = 0.064, 0.007
    for i, d in enumerate([1, 2, 4, 8, 16]):
        bx = tx + 0.012 + i * (bw + gap)
        box(ax, bx, y + 0.012, bw, h - 0.03, f"block {i+1}", f"dilation {d}", tsize=6.6, bsize=6.0)
        if i:
            arrow(ax, bx - gap, y + 0.012 + (h - 0.03) / 2, bx, y + 0.012 + (h - 0.03) / 2)
    ax.text(tx + tw / 2, y + h + 0.003, "block = 2 × (causal conv k=7, 64 ch, ReLU, dropout 0.2) + skip",
            ha="center", va="center", fontsize=5.4, color=MUTED)
    arrow(ax, 0.255, y + h / 2, tx, y + h / 2)
    box(ax, 0.648, y, 0.14, h, "Window score", "time mean → 64-d\nLinear 64 → 2\nP(seizure | window)")
    arrow(ax, tx + tw, y + h / 2, 0.648, y + h / 2)
    box(ax, 0.806, y, 0.189, h, "Clip score", "geometric mean over\nthe clip's windows\n→ P_E(seizure)", color=EEG)
    arrow(ax, 0.788, y + h / 2, 0.806, y + h / 2)
    ax.text(0.005, 0.062, "Receptive field 373 samples (≈3 s) inside each 6 s window. Loss: class-weighted CE on "
            "window labels (ictal if ≥80% of the window lies in the annotated seizure).\nEEG detects seizures well "
            "(macro-F1 ≈0.98) but ranks severity at chance within a session, so the model is trained for seizure vs "
            "non-seizure only.", fontsize=6.0, color=MUTED, va="top", linespacing=1.4)
    save(fig, "fig_eeg_detector")


def fig_egrg():
    fig = plt.figure(figsize=(7.2, 2.9)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.9 / 7.2); ax.axis("off")
    ax.text(0.005, 0.395, "EEG-Gated Racine Grader: EEG and video decide seizure vs non-seizure; video alone grades severity",
            fontsize=8.2, fontweight="bold", color=INK, va="top")
    h = 0.095
    vy, ey = 0.245, 0.085
    box(ax, 0.005, vy, 0.1, h, "Video", "15 fps clip", color=VID)
    box(ax, 0.12, vy, 0.145, h, "Video grader", "X3D-M with 3- and\n5-class heads", color=VID)
    arrow(ax, 0.105, vy + h / 2, 0.12, vy + h / 2)
    # grade split on top, seizure prob below (no crossing)
    box(ax, 0.30, vy + 0.052, 0.235, 0.045, "P_V(grade | seizure), renormalised", tsize=6.4, bold=False, color=VID)
    box(ax, 0.30, vy - 0.002, 0.235, 0.045, "P_V(seizure) = 1 − P_V(non-seizure)", tsize=6.4, bold=False, color=VID)
    arrow(ax, 0.265, vy + h / 2, 0.30, vy + 0.0745)
    arrow(ax, 0.265, vy + h / 2, 0.30, vy + 0.0205)
    box(ax, 0.005, ey, 0.1, h, "EEG", "1 channel\n1 kHz", color=EEG)
    box(ax, 0.12, ey, 0.145, h, "EEG detector", "TCN over 6 s windows,\ngeometric-mean pooling", bsize=5.8, color=EEG)
    arrow(ax, 0.105, ey + h / 2, 0.12, ey + h / 2)
    box(ax, 0.30, ey + 0.025, 0.235, 0.045, "P_E(seizure)", tsize=6.4, bold=False, color=EEG)
    arrow(ax, 0.265, ey + h / 2, 0.30, ey + h / 2)
    gx, gy, gw, gh = 0.575, 0.105, 0.2, 0.15
    box(ax, gx, gy, gw, gh, "Gate, 3 parameters",
        "q = σ(a · logit P_V(sz)\n + b · logit P_E(sz) + c)\n\nlogistic regression fitted\nleave-one-animal-out;\nrefit for each video model",
        color=GATE, fill="#f1fbf6")
    arrow(ax, 0.535, vy + 0.0205, gx, gy + gh - 0.035, color=VID)
    arrow(ax, 0.535, ey + 0.0475, gx, gy + 0.035, color=EEG)
    ox, ow = 0.805, 0.19
    box(ax, ox, 0.105, ow, 0.2, "Output",
        "P(non-seizure) = 1 − q\nP(grade g) = q · P_V(g | sz)\n\ng ∈ {mild, severe}\nor {S2, S3, S4, S5}\n\ndecision: argmax",
        color=GATE, fill="#f1fbf6")
    arrow(ax, gx + gw, gy + gh / 2, ox, gy + gh / 2, label="q")
    arrow(ax, 0.535, vy + 0.0745, ox + ow / 2, 0.305, color=VID, rad=-0.12)
    ax.text(0.69, 0.352, "grade split comes from video only", fontsize=6.0, color=MUTED, ha="center")
    ax.text(0.005, 0.058, "The EEG network is interchangeable inside the gate (GRU, TCN and a small CPU-trained TCN agree "
            "within 0.002 macro-F1). The gain comes from EEG's independent\nseizure evidence (EEG and video detection "
            "errors overlap on 14 clips against 2.4 expected by chance) and from calibrating the two probabilities.",
            fontsize=6.0, color=MUTED, va="top", linespacing=1.4)
    save(fig, "fig_egrg_system")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    fig_video(); fig_eeg(); fig_egrg()
    print("wrote", sorted(f for f in os.listdir(HERE) if f.startswith("fig_")))
