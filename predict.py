import argparse, glob, json, os, sqlite3, sys, time
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, requests
from dotenv import load_dotenv
from model import load, fit, predict, rps

XI, REG = 0.0019, 3.0
COMPS = ["PL", "BL1", "SA", "PPL", "FL1", "PD", "ELC", "CL"]
SEASON = 2026
BASE = "https://api.football-data.org/v4"
ISO = "%Y-%m-%dT%H:%M:%SZ"
MIN_GAMES = 15

def log(*a):
    print(*a, file=sys.stderr)

def refresh():
    load_dotenv()
    h = {"X-Auth-Token": os.getenv("FOOTBALL_DATA_TOKEN")}
    for c in COMPS:
        r = requests.get(f"{BASE}/competitions/{c}/matches?season={SEASON}", headers=h, timeout=30)
        if r.status_code == 200:
            json.dump(r.json()["matches"], open(f"data/{c}_{SEASON}.json", "w"))
            log("refresh OK   ", c)
        else:
            log("refresh GAGAL", c, r.status_code)
        time.sleep(7)

def db():
    con = sqlite3.connect("predictions.db")
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        match_id INTEGER, comp TEXT, home TEXT, away TEXT, kickoff TEXT, generated_at TEXT,
        pH REAL, pD REAL, pA REAL, bH REAL, bD REAL, bA REAL,
        lam_h REAL, lam_a REAL, top_scores TEXT, p_over25 REAL, p_btts REAL,
        hg INTEGER, ag INTEGER, PRIMARY KEY(match_id, generated_at))""")
    return con

def run(days, as_json):
    df = load()
    now = datetime.now(timezone.utc)
    limit = now + timedelta(days=days)
    gen = now.strftime(ISO)
    ref = pd.Timestamp.now(tz="UTC")
    con = db()
    out = []
    for comp in COMPS:
        train = df[df.comp == comp]
        model = fit(train, ref, xi=XI, reg=REG)
        base = [(train.hg > train.ag).mean(), (train.hg == train.ag).mean(), (train.hg < train.ag).mean()]
        cnt = pd.concat([train.home, train.away]).value_counts()
        for m in json.load(open(f"data/{comp}_{SEASON}.json")):
            if m["status"] not in ("SCHEDULED", "TIMED"):
                continue
            ko = datetime.strptime(m["utcDate"], ISO).replace(tzinfo=timezone.utc)
            if not (now <= ko <= limit):
                continue
            home, away = m["homeTeam"]["name"], m["awayTeam"]["name"]
            p, M, lh, la = predict(model, home, away)
            top = [(int(i), int(j), float(M[i, j])) for i, j in
                   sorted(np.ndindex(M.shape), key=lambda ij: -M[ij])[:3]]
            over25 = float(sum(M[i, j] for i, j in np.ndindex(M.shape) if i + j >= 3))
            btts = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
            rec = dict(match_id=m["id"], comp=comp, home=home, away=away, kickoff=m["utcDate"],
                       kickoff_wib=(ko + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M"),
                       p_home=round(float(p[0]), 3), p_draw=round(float(p[1]), 3), p_away=round(float(p[2]), 3),
                       exp_goals=[round(float(lh), 2), round(float(la), 2)],
                       data_terbatas=[t for t in (home, away) if cnt.get(t, 0) < MIN_GAMES],
                       top_scores=[{"score": f"{i}-{j}", "prob": round(pr, 3)} for i, j, pr in top],
                       p_over25=round(over25, 3), p_btts=round(btts, 3))
            out.append(rec)
            con.execute("INSERT OR IGNORE INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                        (m["id"], comp, home, away, m["utcDate"], gen, *map(float, p), *map(float, base),
                         float(lh), float(la), json.dumps(rec["top_scores"]), over25, btts))
    con.commit()
    out.sort(key=lambda r: r["kickoff"])
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    for r in out:
        ts = ", ".join(f'{s["score"]} ({s["prob"]:.0%})' for s in r["top_scores"])
        warn = f' ⚠ data terbatas: {", ".join(r["data_terbatas"])}' if r["data_terbatas"] else ""
        print(f'{r["kickoff_wib"]} WIB {r["comp"]:3s} {r["home"]} vs {r["away"]}{warn}\n'
              f'   H/D/A {r["p_home"]:.0%}/{r["p_draw"]:.0%}/{r["p_away"]:.0%} | skor: {ts} | '
              f'O2.5 {r["p_over25"]:.0%} | BTTS {r["p_btts"]:.0%}')
    print(f"\n{len(out)} prediksi dicatat di predictions.db")

def report():
    con = db()
    for f in glob.glob("data/*.json"):
        for m in json.load(open(f)):
            ft = m["score"]["fullTime"]
            if m["status"] == "FINISHED" and ft["home"] is not None:
                con.execute("UPDATE predictions SET hg=?, ag=? WHERE match_id=? AND hg IS NULL",
                            (ft["home"], ft["away"], m["id"]))
    con.commit()
    rows = con.execute("""SELECT p.pH,p.pD,p.pA,p.bH,p.bD,p.bA,p.hg,p.ag FROM predictions p
        JOIN (SELECT match_id, MAX(generated_at) g FROM predictions
              WHERE hg IS NOT NULL AND generated_at < kickoff GROUP BY match_id) m
        ON p.match_id=m.match_id AND p.generated_at=m.g""").fetchall()
    if not rows:
        print("Belum ada prediksi yang sudah selesai dimainkan.")
        return
    rm, d = [], []
    for pH, pD, pA, bH, bD, bA, hg, ag in rows:
        o = 0 if hg > ag else (1 if hg == ag else 2)
        a, b = rps(np.array([pH, pD, pA]), o), rps(np.array([bH, bD, bA]), o)
        rm.append(a); d.append(b - a)
    d = np.array(d)
    print(f"n={len(rows)}  RPS model={np.mean(rm):.4f}  selisih vs baseline={d.mean():+.4f} "
          f"(±{d.std() / np.sqrt(len(d)):.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--refresh", action="store_true", help="ambil ulang data musim berjalan (8 request)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", action="store_true", help="isi hasil laga selesai & hitung skor")
    a = ap.parse_args()
    if a.refresh:
        refresh()
    report() if a.report else run(a.days, a.json)