import json, glob
for f in sorted(glob.glob("data/*.json")):
    m = json.load(open(f))
    done = sum(1 for x in m if x["status"] == "FINISHED")
    print(f, "total:", len(m), "selesai:", done)