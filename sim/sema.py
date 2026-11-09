"""Sekil 1: mentor karar dongusu diyagrami (matplotlib)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4)
boxes = [("Oyuncunun\neylemi ve\nsüresi", 0.2, "#e8e8e8"), ("Oyuncu modeli\n(bilgi izleme,\nhata, süre,\nyardım geçmişi)", 2.6, "#cfe3f7"),
         ("Bağlam\nvektörü x\n(7 özellik)", 5.0, "#cfe3f7"), ("LinUCB\nmentör\n(4 kol)", 7.4, "#f7dcc4"),
         ("Yardım seviyesi\n0 yok · 1 ipucu\n2 açıklama\n3 doğrudan", 9.8, "#f7dcc4")]
for t, x, c in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.0, 1.6, boxstyle="round,pad=0.05", fc=c, ec="black"))
    ax.text(x + 1.0, 1.8, t, ha="center", va="center", fontsize=8)
for i in range(4):
    x0 = boxes[i][1] + 2.05; ax.add_patch(FancyArrowPatch((x0, 1.8), (boxes[i + 1][1] - 0.05, 1.8), arrowstyle="-|>", mutation_scale=12))
ax.add_patch(FancyArrowPatch((10.8, 0.95), (1.2, 0.95), connectionstyle="arc3,rad=-0.0", arrowstyle="-|>", mutation_scale=12, linestyle="--"))
ax.text(6.0, 0.55, "oyuncunun tepkisi → ödül  r = başarı − λ·(yardım/3) + κ·Δbeceri  → LinUCB güncellemesi", ha="center", fontsize=8)
for ext in ("pdf", "png"):
    plt.savefig(FIG / f"mentor_dongusu.{ext}", bbox_inches="tight", dpi=150)
