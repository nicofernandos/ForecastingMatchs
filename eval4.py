import numpy as np, pandas as pd
from model import load, fit, predict

XI, REG, LOW = 0.0019, 3.0, 15
df = load()
rows = []
for comp in ["PL", "BL1", "SA", "PPL"]:
    dc = df[df.comp == comp]
    for season in (2025, 2026):
        gc = dc[dc.season == season]
        gc = gc.assign(wk=gc["date"].dt.strftime("%G-%V"))
        for _, gw in gc.groupby("wk"):
            ref = gw["date"].min()
            prior = dc[dc["date"] < ref]
            cnt = pd.concat([prior.home, prior.away]).value_counts()
            m = fit(prior, ref, xi=XI, reg=REG)
            for _, r in gw.iterrows():
                p, *_ = predict(m, r["home"], r["away"])
                rows.append(dict(
                    season=season, nh=cnt.get(r["home"], 0), na=cnt.get(r["away"], 0),
                    pred_h=3 * p[0] + p[1], pred_a=3 * p[2] + p[1],
                    act_h=3 if r.hg > r.ag else (1 if r.hg == r.ag else 0),
                    act_a=3 if r.ag > r.hg else (1 if r.hg == r.ag else 0)))
res = pd.DataFrame(rows)
long = pd.concat([
    res.assign(n=res.nh, n_opp=res.na, pred=res.pred_h, act=res.act_h),
    res.assign(n=res.na, n_opp=res.nh, pred=res.pred_a, act=res.act_a)])

def summ(name, g):
    d = g.act - g.pred
    print(f"{name:34s} n={len(g):4d}  poin/laga prediksi={g.pred.mean():.2f} "
          f"aktual={g.act.mean():.2f}  selisih={d.mean():+.2f} (±{d.std() / np.sqrt(len(g)):.2f})")

low = (long.n < LOW) & (long.n_opp >= LOW)
ok = (long.n >= LOW) & (long.n_opp >= LOW)
summ("tim data sedikit (<15 laga)", long[low])
summ("  - hanya musim 2025", long[low & (long.season == 2025)])
summ("  - hanya musim 2026", long[low & (long.season == 2026)])
summ("tim data cukup vs data cukup", long[ok])