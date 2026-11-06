import numpy as np, pandas as pd
from model import load, fit, predict, rps

def elo_probs(df, K=20, HA=65):
    """Elo berjalan kronologis per liga; prediksi dicatat SEBELUM update."""
    out = {}
    for comp, g in df.groupby("comp"):
        g = g.sort_values("date")
        R, n_all, n_draw = {}, 0, 0
        for _, r in g.iterrows():
            rh, ra = R.get(r["home"], 1500.0), R.get(r["away"], 1500.0)
            E = 1 / (1 + 10 ** (-(rh + HA - ra) / 400))
            dr = (n_draw / n_all) if n_all >= 30 else 0.26
            pD = dr * 4 * E * (1 - E)
            p = np.array([E - pD / 2, pD, 1 - E - pD / 2])
            out[(comp, r["home"], r["away"], r["date"])] = p
            o = 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)
            S = (1.0, 0.5, 0.0)[o]
            R[r["home"]] = rh + K * (S - E)
            R[r["away"]] = ra - K * (S - E)
            n_all += 1; n_draw += (o == 1)
    return out

if __name__ == "__main__":
    df = load()
    elo = elo_probs(df)
    rows = []
    test = df[df.season == 2026]
    for comp, gc in test.groupby("comp"):
        for _, gd in gc.groupby(gc["date"].dt.date):
            ref = gd["date"].min()
            m = fit(df[(df.comp == comp) & (df["date"] < ref)], ref)
            for _, r in gd.iterrows():
                pm, *_ = predict(m, r["home"], r["away"])
                pe = elo[(comp, r["home"], r["away"], r["date"])]
                o = 0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2)
                rows.append(dict(comp=comp, o=o,
                                 rps_m=rps(pm, o), rps_e=rps(pe, o),
                                 pH=pm[0], pD=pm[1], pA=pm[2]))
    res = pd.DataFrame(rows)

    print("== Model Poisson vs Elo (selisih positif = Poisson lebih baik) ==")
    for name, g in list(res.groupby("comp")) + [("SEMUA", res)]:
        d = g.rps_e - g.rps_m
        print(f"{name:6s} n={len(g):3d}  RPS poisson={g.rps_m.mean():.4f} "
              f"elo={g.rps_e.mean():.4f}  selisih={d.mean():+.4f} (±{d.std()/np.sqrt(len(g)):.4f})")

    print("\n== Kalibrasi Poisson, semua laga 2026 ==")
    for k, lab in enumerate(["Menang kandang", "Seri", "Menang tandang"]):
        pr = res[["pH", "pD", "pA"]].iloc[:, k].mean()
        ac = (res.o == k).mean()
        print(f"{lab:15s} prediksi rata-rata={pr:.3f}  frekuensi aktual={ac:.3f}")