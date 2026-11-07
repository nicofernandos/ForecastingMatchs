import os, json, time, requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
HEADERS = {"X-Auth-Token": TOKEN}
BASE = "https://api.football-data.org/v4"

COMPETITIONS = ["PL", "BL1", "SA", "PPL", "FL1", "PD", "ELC", "CL"]
SEASONS = [2024, 2025, 2026]  # 2025 = musim 2025/26

for comp in COMPETITIONS:
    for season in SEASONS:
        url = f"{BASE}/competitions/{comp}/matches?season={season}"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            matches = r.json().get("matches", [])
            with open(f"data/{comp}_{season}.json", "w") as f:
                json.dump(matches, f)
            print(f"OK   {comp} {season}: {len(matches)} laga")
        else:
            print(f"GAGAL {comp} {season}: HTTP {r.status_code} {r.text[:120]}")
        time.sleep(7)  # jaga batas 10 request/menit