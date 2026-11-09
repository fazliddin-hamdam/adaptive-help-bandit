"""Analiz: CSV -> tohum duzeyinde ortalama, %95 GA, Wilcoxon, Cohen d, grafik ve LaTeX tablo."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB, FIG = RES / "tables", RES / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

def save(name):
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"{name}.{ext}", dpi=150)
    plt.close()

df = pd.read_csv(RES / "test_sonuclari.csv")
M = ["basari", "hata", "ipucu", "bagimlilik", "ogrenme", "sure", "beceri_tahmin"]
seed_df = df.groupby(["politika", "profil", "zorluk", "seed"])[M].mean().reset_index()   # tohum duzeyi (n=10)

def agg(sub, k):
    v = sub[k].values; n = len(v)
    ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean(), v.std(ddof=1), ci

def compare(a, b, prof, diff, k):
    f = lambda p: seed_df[(seed_df.politika == p) & (seed_df.profil == prof) & (seed_df.zorluk == diff)].sort_values("seed")[k].values
    x, y = f(a), f(b); d = x - y
    w = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return d.mean(), w, d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0

# --- Ozet: orta zorluk, tum 3 profil birlikte (profil ortalamasi)
prof3 = ["hizli", "orta", "yeni"]
names = {"P0": "P0 Sabit", "P1": "P1 Modelsiz", "P2": "P2 Tam", "P3": "P3 -beceri", "P4": "P4 -hata",
         "P5": "P5 -sure", "P6": "P6 -yardim", "P7": "P7 LinTS", "S0": "S0 Hep 0", "S3": "S3 Hep 3"}
lines = []
for diff in ["orta"]:
    for p in names:
        sub = seed_df[(seed_df.politika == p) & seed_df.profil.isin(prof3) & (seed_df.zorluk == diff)] \
            .groupby("seed")[M].mean().reset_index()
        cells = []
        for k in ["basari", "hata", "ipucu", "bagimlilik", "ogrenme"]:
            mu, sd, ci = agg(sub, k); cells.append(f"{mu:.2f} $\\pm$ {sd:.2f}")
        lines.append(f"{names[p]} & " + " & ".join(cells) + " \\\\")
open(TAB / "tablo_ozet.tex", "w", encoding="utf8").write("\n".join(lines))

# --- Profil x zorluk: P0, P1, P2 basari/ogrenme ve Wilcoxon
rows = []
for prof in prof3:
    for diff in ["dusuk", "orta", "yuksek"]:
        for k in ["basari", "ogrenme"]:
            mu = {p: agg(seed_df[(seed_df.politika == p) & (seed_df.profil == prof) & (seed_df.zorluk == diff)], k) for p in ["P0", "P1", "P2"]}
            _, w20, d20 = compare("P2", "P0", prof, diff, k)
            _, w21, d21 = compare("P2", "P1", prof, diff, k)
            rows.append(dict(profil=prof, zorluk=diff, metrik=k, P0=mu["P0"][0], P1=mu["P1"][0], P2=mu["P2"][0],
                             ci_P2=mu["P2"][2], p_P2_P0=w20, d_P2_P0=d20, p_P2_P1=w21, d_P2_P1=d21))
res = pd.DataFrame(rows); res.to_csv(RES / "karsilastirma.csv", index=False)
lines = []
for _, r in res[res.metrik == "basari"].iterrows():
    dd = "$>$10" if r.d_P2_P0 > 10 else format(r.d_P2_P0, ".1f")
    lines.append(f"{r.profil} & {r.zorluk} & {r.P0:.2f} & {r.P1:.2f} & {r.P2:.2f} $\\pm$ {r.ci_P2:.2f} & {r.p_P2_P0:.3f} & {dd} \\\\")
open(TAB / "tablo_profil.tex", "w", encoding="utf8").write("\n".join(lines))

# --- Stres testi (orta zorluk)
lines = []
for lab in ["orta", "yorulan", "gurultu3x"]:
    cells = []
    for p in ["P0", "P1", "P2", "S3"]:
        mu, sd, ci = agg(seed_df[(seed_df.politika == p) & (seed_df.profil == lab) & (seed_df.zorluk == "orta")], "basari")
        cells.append(f"{mu:.2f} $\\pm$ {sd:.2f}")
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
open(TAB / "tablo_stres.tex", "w", encoding="utf8").write("\n".join(lines))

# --- Grafikler
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
for a, prof in zip(ax, prof3):
    for p, c in [("P0", "tab:gray"), ("P1", "tab:orange"), ("P2", "tab:blue")]:
        mu = [agg(seed_df[(seed_df.politika == p) & (seed_df.profil == prof) & (seed_df.zorluk == d)], "basari") for d in ["dusuk", "orta", "yuksek"]]
        a.errorbar(range(3), [x[0] for x in mu], yerr=[x[2] for x in mu], marker="o", capsize=3, label=names[p], color=c)
    a.set_xticks(range(3)); a.set_xticklabels(["düşük", "orta", "yüksek"]); a.set_title({"hizli": "hızlı"}.get(prof, prof)); a.set_xlabel("zorluk")
ax[0].set_ylabel("Görev başarısı"); ax[0].legend(fontsize=7)
plt.tight_layout(); save("basari_zorluk")

fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for a, k, t in zip(ax, ["bagimlilik", "ogrenme"], ["Yardıma bağımlılık", "Gerçek öğrenme kazancı"]):
    vals = []
    for p in names:
        sub = seed_df[(seed_df.politika == p) & seed_df.profil.isin(prof3) & (seed_df.zorluk == "orta")].groupby("seed")[[k]].mean().reset_index()
        vals.append(agg(sub, k))
    a.bar(range(len(names)), [v[0] for v in vals], yerr=[v[2] for v in vals], capsize=2, color="steelblue")
    a.set_xticks(range(len(names))); a.set_xticklabels(list(names), rotation=45, fontsize=7); a.set_title(t)
plt.tight_layout(); save("ablation")
print(open(TAB / "tablo_ozet.tex", encoding="utf8").read()); print(); print(open(TAB / "tablo_profil.tex", encoding="utf8").read()); print(); print(open(TAB / "tablo_stres.tex", encoding="utf8").read())


