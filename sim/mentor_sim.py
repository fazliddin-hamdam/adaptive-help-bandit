"""Pilot simulasyon: sentetik ogrenen, oyuncu modeli (BKT) ve LinUCB / LinTS mentor politikalari.
Harici ML kutuphanesi kullanilmaz; yalnizca numpy ve scipy.
"""
import numpy as np
from scipy.special import expit

CATS = 5                                   # beceri kategorisi (her biri 2 senaryo)
N_SCEN = 10                                # senaryo sayisi
HELP_BOOST = np.array([0.0, 0.25, 0.55, 0.95])   # yardim seviyesinin basari olasiligina etkisi
LEARN_MULT = np.array([1.0, 0.9, 0.6, 0.2])      # yuksek yardim = dusuk ogrenme (uretken caba)
PROFILES = {                               # sentetik ogrenen profilleri
    "hizli": dict(s0=0.70, lr=0.12, dep0=0.05),
    "orta":  dict(s0=0.50, lr=0.08, dep0=0.20),
    "yeni":  dict(s0=0.25, lr=0.05, dep0=0.45),
}
DIFFS = {                                  # zorluk seviyeleri: adim sayisi, secenek sayisi, zorluk kaymasi
    "dusuk":  dict(steps=4, opts=3, shift=0.00),
    "orta":   dict(steps=6, opts=4, shift=0.10),
    "yuksek": dict(steps=8, opts=5, shift=0.20),
}
MAX_ATT = 4                                # bir adimda en fazla 4 deneme (yardim 0..3)

# ---------------------------------------------------------------- sentetik ogrenen
class Learner:
    def __init__(self, profile, rng, tired=False, tnoise=0.1):
        p = PROFILES[profile]
        self.s = np.clip(p["s0"] + rng.normal(0, 0.08, CATS), 0.02, 0.98)  # gizli beceri
        self.s_start = self.s.mean()
        self.lr = p["lr"] * rng.uniform(0.8, 1.2)                          # ogrenme hizi
        self.dep = p["dep0"]                                               # gizli yardim bagimliligi
        self.speed = rng.uniform(0.8, 1.2)
        self.tired, self.tnoise = tired, tnoise

    def p_unassisted(self, c, d, opts):
        base = expit(6 * (self.s[c] - d))                  # beceri - zorluk farkinin sigmoid'i
        p0 = 1 / opts + (1 - 1 / opts) * base              # sans ile tahmin tabani
        return p0 * (1 - 0.3 * self.dep)                   # bagimlilik yardimsiz basariyi dusurur

    def attempt(self, c, d, opts, h, u, z):
        p0 = self.p_unassisted(c, d, opts)
        p = p0 + (1 - p0) * HELP_BOOST[h]
        ok = u < p
        t = (1 + 0.6 * self.dep) * (1 + 0.8 * (1 - p0)) * self.speed * (1 + self.tnoise * z)
        # ogrenme
        gain = self.lr * (1 - self.s[c]) * (LEARN_MULT[h] if ok else 0.15)
        self.s[c] = min(0.99, self.s[c] + gain)
        self.dep = min(1.0, self.dep + 0.04 * h) if h >= 1 else max(0.0, self.dep - 0.03)
        return ok, max(t, 0.2), p0

    def get_tired(self):                                   # stres testi: oturum ortasinda dusus
        self.s = np.clip(self.s - 0.15, 0.02, 0.98)
        self.speed *= 1.3

# ---------------------------------------------------------------- oyuncu modeli (BKT + ozellikler)
class PlayerModel:
    P_INIT, P_LEARN, SLIP, GUESS = 0.3, 0.15, 0.1, 0.2

    def __init__(self):
        self.L = np.full(CATS, self.P_INIT)                # beceri bilme olasiligi
        self.err, self.rt = 0.0, 0.5
        self.hist = []                                     # son yardim seviyeleri

    def update(self, c, ok, h, t):
        g = min(0.9, self.GUESS + 0.2 * h)                 # yardimla tahmin olasiligi artar
        L = self.L[c]
        post = L * (1 - self.SLIP) / (L * (1 - self.SLIP) + (1 - L) * g) if ok \
            else L * self.SLIP / (L * self.SLIP + (1 - L) * (1 - g))
        new = post + (1 - post) * self.P_LEARN
        delta, self.L[c] = new - L, new
        self.err = 0.7 * self.err + 0.3 * (0 if ok else 1)
        self.rt = min(max(t / 1.5, 0.0), 2.0) / 2.0
        self.hist = (self.hist + [h])[-5:]
        return delta

    def context(self, c, d, wrong):
        hh = np.mean([x >= 1 for x in self.hist]) if self.hist else 0.0
        return np.array([1.0, self.L[c], self.err, self.rt, hh, d, wrong / 3])

# ---------------------------------------------------------------- politikalar
class FixedPolicy:                                         # P0: 1. yanlista ipucu, 2. aciklayici, 3. dogrudan
    learns = False
    def act(self, x, wrong): return min(wrong, 3)
    def update(self, *a): pass

class ConstPolicy:                                         # S0 / S3 sinir politikalari
    learns = False
    def __init__(self, h): self.h = h
    def act(self, x, wrong): return self.h
    def update(self, *a): pass

class LinUCB:                                              # P1, P2, P3-P6 (maskeyle)
    learns = True
    def __init__(self, mask, alpha=0.3, d=7):
        self.m = np.array(mask); k = len(mask)
        self.Ainv = [np.eye(k) for _ in range(4)]          # A_a^-1 (Sherman-Morrison ile tutulur)
        self.b = [np.zeros(k) for _ in range(4)]
        self.alpha = alpha
    def act(self, x, wrong):
        xm = x[self.m]
        p = [(Ai @ bb) @ xm + self.alpha * np.sqrt(max(xm @ Ai @ xm, 0)) for Ai, bb in zip(self.Ainv, self.b)]
        return int(np.argmax(p))
    def update(self, x, a, r):
        xm = x[self.m]; Ai = self.Ainv[a]; Ax = Ai @ xm
        self.Ainv[a] = Ai - np.outer(Ax, Ax) / (1 + xm @ Ax)   # (A + x x^T)^-1
        self.b[a] = self.b[a] + r * xm

class LinTS(LinUCB):                                       # P7: dogrusal Thompson ornekleme
    def __init__(self, mask, alpha=0.3, seed=0):
        super().__init__(mask, alpha); self.rng = np.random.default_rng(seed)
    def act(self, x, wrong):
        xm = x[self.m]; best, ba = -1e9, 0
        for a in range(4):
            Ai = self.Ainv[a]
            th = self.rng.multivariate_normal(Ai @ self.b[a], (self.alpha ** 2) * Ai)
            if th @ xm > best: best, ba = th @ xm, a
        return ba

FULL = [0, 1, 2, 3, 4, 5, 6]
MASKS = {
    "P1": [0, 5, 6],                                       # oyuncu modelsiz
    "P2": FULL,
    "P3": [0, 2, 3, 4, 5, 6],                              # beceri cikarildi
    "P4": [0, 1, 3, 4, 5, 6],                              # son hatalar cikarildi
    "P5": [0, 1, 2, 4, 5, 6],                              # tepki suresi cikarildi
    "P6": [0, 1, 2, 3, 5, 6],                              # yardim gecmisi cikarildi
}
def make_policy(name, alpha=0.3, seed=0):
    if name == "P0": return FixedPolicy()
    if name == "S0": return ConstPolicy(0)
    if name == "S3": return ConstPolicy(3)
    if name == "P7": return LinTS(MASKS["P2"], alpha, seed)
    return LinUCB(MASKS[name], alpha)

# ---------------------------------------------------------------- bir oturum (10 senaryo)
def step_difficulty(sc, st, shift):
    return float(np.clip(0.2 + 0.6 * ((sc * 7 + st * 3) % 10) / 9 + shift, 0.05, 0.95))

def run_session(policy, learner, diff, U, Z, lam=0.3, kappa=1.0, update=False):
    cfg = DIFFS[diff]; pm = PlayerModel()
    errs = succ = hints = att_n = time_sum = 0.0
    dep_help = dep_tot = 0
    for sc in range(N_SCEN):
        if learner.tired and sc == 5: learner.get_tired()
        c, sc_err = sc // 2, 0
        for st in range(cfg["steps"]):
            d = step_difficulty(sc, st, cfg["shift"])
            for wrong in range(MAX_ATT):
                x = pm.context(c, d, wrong)
                h = policy.act(x, wrong)
                ok, t, p0 = learner.attempt(c, d, cfg["opts"], h, U[sc, st, wrong], Z[sc, st, wrong])
                delta = pm.update(c, ok, h, t)
                if update and policy.learns:
                    policy.update(x, h, float(ok) - lam * h / 3 + kappa * delta)
                time_sum += t; att_n += 1; hints += (h >= 1)
                if p0 >= 0.7:
                    dep_tot += 1; dep_help += (h >= 1)
                if ok: break
                sc_err += 1
        errs += sc_err
        succ += sc_err <= int(np.ceil(cfg["steps"] / 2))   # hata siniri asilmadan bitirme
    return dict(basari=succ / N_SCEN, sure=time_sum / N_SCEN, hata=errs / N_SCEN,
                ipucu=hints / N_SCEN, beceri_tahmin=float(pm.L.mean()),
                ogrenme=float(learner.s.mean() - learner.s_start),
                bagimlilik=(dep_help / dep_tot) if dep_tot else 0.0)

def make_episode(seed, ep, profile, diff, tired=False, tnoise=0.1, pname=None):
    """Ortak rastgele sayilar: ayni (seed, ep, profil) icin tum politikalar ayni ogrenen ve zar atislarini gorur."""
    key = [seed, ep, list(PROFILES).index(profile), list(DIFFS).index(diff)]
    L = Learner(profile, np.random.default_rng(key + [0]), tired, tnoise)
    r = np.random.default_rng(key + [1])
    return L, r.random((N_SCEN, 8, MAX_ATT)), r.standard_normal((N_SCEN, 8, MAX_ATT))

def train(name, diff, alpha, lam, kappa, n_ep=150):
    pol = make_policy(name, alpha, seed=1)
    if not pol.learns: return pol
    for ep in range(n_ep):                                   # egitim tohumlari 0-99
        prof = list(PROFILES)[ep % 3]
        L, U, Z = make_episode(ep % 100, ep, prof, diff)
        run_session(pol, L, diff, U, Z, lam, kappa, update=True)
    if hasattr(pol, "alpha"): pol.alpha = 0.0 if name != "P7" else pol.alpha   # test: agirliklar donuk, acgozlu
    return pol

