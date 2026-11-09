"""Deney kosucusu: dogrulama (hiperparametre) -> test (tohum 1000-1009) -> CSV."""
import csv, itertools, json, time
import numpy as np
import mentor_sim as m
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

POLICIES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "S0", "S3"]
SEEDS_TEST = range(1000, 1010)          # test tohumlari
SEEDS_VAL = range(100, 110)             # dogrulama tohumlari
N_EP = 30
N_TRAIN = 300
RES.mkdir(exist_ok=True)

def evaluate(pol, diff, profile, seeds, n_ep, tired=False, tnoise=0.1):
    rows = []
    for s in seeds:
        for ep in range(n_ep):
            L, U, Z = m.make_episode(s, ep, profile, diff, tired, tnoise)
            rows.append(dict(seed=s, episode=ep, **m.run_session(pol, L, diff, U, Z)))
    return rows

# 1) Dogrulama: her ogrenen politika icin alpha, lambda, kappa yalnizca dogrulama tohumlarinda secilir (orta zorluk).
#    Secim olcutu deneyden ONCE sabitlendi: basari + 2*ogrenme - 0.5*bagimlilik
LEARNERS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
HP, grid = {}, []
for name in LEARNERS:
    best = None
    for alpha, lam, kappa in itertools.product([0.2, 0.6], [0.1, 0.3, 0.6], [0.5, 1.0, 2.0]):
        pol = m.train(name, "orta", alpha, lam, kappa, N_TRAIN)
        r = [x for prof in m.PROFILES for x in evaluate(pol, "orta", prof, SEEDS_VAL, 3)]
        mean = lambda k: float(np.mean([x[k] for x in r]))
        score = mean("basari") + 2 * mean("ogrenme") - 0.5 * mean("bagimlilik")
        grid.append(dict(politika=name, alpha=alpha, lam=lam, kappa=kappa, score=round(score, 4)))
        if best is None or score > best[0]: best = (score, alpha, lam, kappa)
    HP[name] = best[1:]
    print(name, "secilen (alpha, lam, kappa):", best[1:], "skor", round(best[0], 3))
json.dump(dict(secilen={k: dict(alpha=v[0], lam=v[1], kappa=v[2]) for k, v in HP.items()}, grid=grid),
          open(RES / "validation.json", "w"), indent=1)
# 2) Test: profil x zorluk x politika, 10 tohum x 30 episode; politikalar donuk
conds = [(p, d, False, 0.1, p) for p in m.PROFILES for d in m.DIFFS]
conds += [("orta", "orta", True, 0.1, "yorulan"), ("orta", "orta", False, 0.3, "gurultu3x")]   # stres testleri
out = []
t0 = time.time()
for diff in m.DIFFS:
    pols = {n: m.train(n, diff, *HP.get(n, (0.3, 0.3, 1.0)), N_TRAIN) for n in POLICIES}
    for prof, d, tired, tn, label in conds:
        if d != diff: continue
        for n, pol in pols.items():
            for r in evaluate(pol, diff, prof, SEEDS_TEST, N_EP, tired, tn):
                out.append(dict(politika=n, profil=label, zorluk=diff, **r))
    print(diff, "bitti", round(time.time() - t0), "sn")
with open(RES / "test_sonuclari.csv", "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("satir:", len(out))

