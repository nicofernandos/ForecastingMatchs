import json, glob, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

def load():
    rows = []
    for f in glob.glob("data/*.json"):
        comp, season = os.path.basename(f)[:-5].split("_")
        for m in json.load(open(f)):
            if m["status"] != "FINISHED":
                continue
            ft = m["score"]["fullTime"]
            if ft["home"] is None:
                continue
            rows.append(dict(comp=comp, season=int(season),
                             date=pd.to_datetime(m["utcDate"]),
                             home=m["homeTeam"]["name"], away=m["awayTeam"]["name"],
                             hg=ft["home"], ag=ft["away"]))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def fit(train, ref_date, xi=0.0019, reg=0.3):
    if train.empty:
        raise ValueError("Data pelatihan kosong.")

    teams = sorted(set(train.home) | set(train.away))
    # kode berikutnya tetap
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = train.home.map(idx).values
    ai = train.away.map(idx).values
    hg, ag = train.hg.values, train.ag.values
    w = np.exp(-xi * (ref_date - train.date).dt.total_seconds().values / 86400)

    def f(p):
        mu, h = p[0], p[1]
        a, d = p[2:2 + n], p[2 + n:]
        eh = mu + h + a[hi] - d[ai]
        ea = mu + a[ai] - d[hi]
        lh, la = np.exp(eh), np.exp(ea)
        nll = np.sum(w * (lh - hg * eh + la - ag * ea)) + reg * (np.sum(a**2) + np.sum(d**2))
        gh, gaw = w * (lh - hg), w * (la - ag)
        g = np.zeros_like(p)
        g[0] = gh.sum() + gaw.sum()
        g[1] = gh.sum()
        ga_ = np.zeros(n); np.add.at(ga_, hi, gh); np.add.at(ga_, ai, gaw)
        gd_ = np.zeros(n); np.add.at(gd_, ai, -gh); np.add.at(gd_, hi, -gaw)
        g[2:2 + n] = ga_ + 2 * reg * a
        g[2 + n:] = gd_ + 2 * reg * d
        return nll, g

    p0 = np.zeros(2 + 2 * n); p0[0] = 0.2
    result = minimize(f, p0, jac=True, method="L-BFGS-B")

    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Fit Poisson gagal: {result.message}")

    p = result.x
    return dict(idx=idx, mu=p[0], h=p[1], a=p[2:2 + n], d=p[2 + n:])

def predict(model, home, away, maxg=8):
    i, j = model["idx"].get(home), model["idx"].get(away)
    a_h = model["a"][i] if i is not None else 0.0   # tim baru: dianggap rata-rata
    d_h = model["d"][i] if i is not None else 0.0
    a_a = model["a"][j] if j is not None else 0.0
    d_a = model["d"][j] if j is not None else 0.0
    lh = np.exp(model["mu"] + model["h"] + a_h - d_a)
    la = np.exp(model["mu"] + a_a - d_h)
    g = np.arange(maxg + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M /= M.sum()
    p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])  # H, D, A
    return p, M, lh, la

def rps(p, o):
    y = np.zeros(3); y[o] = 1
    return 0.5 * ((p[0] - y[0])**2 + (p[0] + p[1] - y[0] - y[1])**2)

def backtest(df):
    out = []
    test = df[df.season == 2026]
    for comp, gc in test.groupby("comp"):
        for _, gd in gc.groupby(gc.date.dt.date):
            ref = gd.date.min()
            train = df[(df.comp == comp) & (df.date < ref)]
            m = fit(train, ref)
            base = np.array([(train.hg > train.ag).mean(),
                             (train.hg == train.ag).mean(),
                             (train.hg < train.ag).mean()])
            for _, r in gd.iterrows():
                p, *_ = predict(m, r.home, r.away)
                o = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
                out.append(dict(comp=comp,
                                rps_m=rps(p, o), rps_b=rps(base, o),
                                ll_m=-np.log(p[o]), ll_b=-np.log(base[o])))
    return pd.DataFrame(out)

if __name__ == "__main__":
    df = load()
    print("Laga selesai dimuat:", len(df))
    res = backtest(df)
    for name, g in list(res.groupby("comp")) + [("SEMUA", res)]:
        d = g.rps_b - g.rps_m   # positif = model lebih baik dari baseline
        print(f"{name:6s} n={len(g):3d}  RPS model={g.rps_m.mean():.4f} "
              f"baseline={g.rps_b.mean():.4f}  selisih={d.mean():+.4f} "
              f"(±{d.std()/np.sqrt(len(g)):.4f})  "
              f"logloss model={g.ll_m.mean():.3f} baseline={g.ll_b.mean():.3f}")