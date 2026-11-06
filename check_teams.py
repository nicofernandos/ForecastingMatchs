import pandas as pd
from model import load, fit

df = load()
ref = pd.Timestamp.now(tz="UTC")
for comp in ["PL", "BL1", "SA", "PPL"]:
    tr = df[df.comp == comp]
    m = fit(tr, ref, xi=0.0019, reg=3.0)
    cnt = pd.concat([tr.home, tr.away]).value_counts()
    rows = sorted(((m["a"][i] + m["d"][i], m["a"][i], m["d"][i], t, cnt.get(t, 0))
                   for t, i in m["idx"].items()), reverse=True)
    print(f"\n== {comp} ==  (rating = serang + bertahan; positif = lebih kuat)")
    for r, a, d, t, c in rows:
        print(f"{r:+.2f}  serang={a:+.2f} bertahan={d:+.2f}  {t:35s} laga={c}")