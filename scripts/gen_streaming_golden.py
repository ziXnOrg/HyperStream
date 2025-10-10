#!/usr/bin/env python3
# Minimal generator for tests/golden/streaming_events.ndjson using a fixed spec/seed.
# This mirrors the committed canonical file to simplify maintenance.

import json
import random
from pathlib import Path

EV_COUNT = 64
START_TS = 1733856000000

SYMS = ["wake","move","rest","turn","hold","pause","stop","go"]
LABELS = ["active","idle"]

random.seed(20251010)

out = []
seq = 1
src_cycle = ["A","B"]
si = 0
for i in range(EV_COUNT):
    src = src_cycle[si % 2]
    si += 1
    kind_r = i % 4
    if kind_r == 0:
        kind = "symbol"
        payload = {"sym": random.choice(SYMS)}
    elif kind_r == 1:
        kind = "numeric"
        payload = {"val": round(random.uniform(-10.0, 100.0), 3)}
    elif kind_r == 2:
        kind = "vector"
        n = 1 + (i % 6)
        vec = [round(random.uniform(-10.0, 10.0), 3) for _ in range(n)]
        payload = {"vec": vec}
    else:
        kind = "label"
        payload = {"label": LABELS[(i//8) % 2]}
    rec = {
        "v": 1,
        "seq": seq,
        "src": src,
        "eid": f"{src}-{seq:04d}",
        "kind": kind,
        "ts_ms": START_TS + i * 10,
        "payload": payload,
    }
    out.append(rec)
    seq += 1

p = Path(__file__).resolve().parents[1] / "tests" / "golden" / "streaming_events.ndjson"
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("w", encoding="utf-8") as f:
    for rec in out:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
print(f"wrote {len(out)} events to {p}")

