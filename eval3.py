import itertools, numpy as np
from model import load, fit, predict, rps
from eval2 import elo_probs

def outcome(r):
    return 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)

def key(r):
    return (r["comp"], r["home"], r["away"], r["date"])

def poisson_probs(df, season, xi, reg):
    """Refit mingguan; hanya memakai laga sebelum minggu itu."""
    out = {}
    test = df[df.season == season]
    for comp, gc in test.groupby("comp"):
        gc = gc.assign(wk=gc["date"].dt.strftime("%G-%V"))
        for _, gw in gc.groupby("wk"):
            ref = gw["date"].min()
            m = fit(df[(df.comp == comp) & (df["date"] < ref)], ref, xi=xi, reg=reg)
            for _, r in gw.iterrows():
                p, *_ = predict(m, r["home"], r["away"])
                out[key(r)] = p
    return out

def scores(df, season, probs):
    t = df[df.season == season]
    return np.array([rps(probs[key(r)], outcome(r)) for _, r in t.iterrows()])

if __name__ == "__main__":
    df = load()

    print("== Tuning di musim 2025 (2026 tidak disentuh) ==")
    best_p = None
    for xi, reg in itertools.product([0.0007, 0.0019, 0.004], [0.1, 0.3, 1.0, 3.0]):
        s = scores(df, 2025, poisson_probs(df, 2025, xi, reg)).mean()
        print(f"Poisson xi={xi} reg={reg}: RPS 2025 = {s:.4f}")
        if best_p is None or s < best_p[0]:
            best_p = (s, xi, reg)

    best_e = None
    for K, HA in itertools.product([10, 20, 30, 40], [50, 65, 80]):
        s = scores(df, 2025, elo_probs(df, K, HA)).mean()
        print(f"Elo K={K} HA={HA}: RPS 2025 = {s:.4f}")
        if best_e is None or s < best_e[0]:
            best_e = (s, K, HA)

    _, xi, reg = best_p
    _, K, HA = best_e
    print(f"\nTerpilih: Poisson xi={xi} reg={reg} | Elo K={K} HA={HA}")

    pp = poisson_probs(df, 2026, xi, reg)
    ep = elo_probs(df, K, HA)
    t = df[df.season == 2026]
    o = [outcome(r) for _, r in t.iterrows()]
    ks = [key(r) for _, r in t.iterrows()]
    r_p = np.array([rps(pp[k], y) for k, y in zip(ks, o)])
    r_e = np.array([rps(ep[k], y) for k, y in zip(ks, o)])
    r_m = np.array([rps((pp[k] + ep[k]) / 2, y) for k, y in zip(ks, o)])

    print("\n== Uji sekali di 2026 (parameter sudah dibekukan) ==")
    print(f"n={len(o)}  RPS poisson={r_p.mean():.4f}  elo={r_e.mean():.4f}  rata-rata={r_m.mean():.4f}")
    for nama, a in [("poisson", r_p), ("rata-rata", r_m)]:
        d = r_e - a
        print(f"{nama} vs elo: selisih={d.mean():+.4f} (±{d.std()/np.sqrt(len(d)):.4f})  [positif = lebih baik dari Elo]")