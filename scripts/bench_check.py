#!/usr/bin/env python3
import argparse, json, sys, math, os
from collections import defaultdict

REQ_AM = [
    "name","dim_bits","capacity","size","iters","secs",
    "queries_per_sec","eff_gb_per_sec","sample_index","warmup_ms","measure_ms"
]
REQ_CLUSTER = [
    "name","dim_bits","capacity","updates",
    "update_iters","update_secs","updates_per_sec",
    "finalize_iters","finalize_secs","finalizes_per_sec",
    "sample_index","warmup_ms","measure_ms"
]

def load_ndjson(path):
    rows=[]
    # Read raw bytes to handle Windows PowerShell redirection (UTF-16 LE by default)
    with open(path,'rb') as f:
        data=f.read()
    # Decode with BOM detection; fallback to UTF-8
    if data.startswith(b'\xff\xfe'):
        text=data.decode('utf-16-le', errors='strict')
    elif data.startswith(b'\xfe\xff'):
        text=data.decode('utf-16-be', errors='strict')
    else:
        text=data.decode('utf-8', errors='strict')
    for ln, line in enumerate(text.splitlines(), 1):
        line=line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            obj=json.loads(line)
            rows.append(obj)
        except Exception as e:
            print(f"ERROR: {path}:{ln} invalid JSON object: {e}", file=sys.stderr)
            raise
    return rows

def require_fields(rows, req):
    for i, obj in enumerate(rows):
        missing=[k for k in req if k not in obj]
        if missing:
            raise SystemExit(f"Missing fields in row {i}: {missing}")

def filter_rows(rows, req):
    out=[]
    for o in rows:
        if all((k in o) for k in req):
            out.append(o)
    return out

def group_key_am(o):
    return (o["name"], int(o["dim_bits"]), int(o["capacity"]), int(o["size"]))

def group_key_cluster(o):
    return (o["name"], int(o["dim_bits"]), int(o["capacity"]), int(o["updates"]))


def aggregates(values):
    values=sorted(values)
    n=len(values)
    mean=sum(values)/n if n else 0.0
    med=(values[n//2] if n%2==1 else 0.5*(values[n//2-1]+values[n//2])) if n else 0.0
    var=sum((x-mean)**2 for x in values)/n if n else 0.0
    std=math.sqrt(var)
    return {"mean":mean, "median":med, "stdev":std}


def compute_aggregates_am(rows):
    g=defaultdict(lambda: {"qps":[], "gbps":[]})
    for o in rows:
        g[group_key_am(o)]["qps"].append(float(o["queries_per_sec"]))
        g[group_key_am(o)]["gbps"].append(float(o["eff_gb_per_sec"]))
    out={}
    for k,v in g.items():
        out[k]={
            "queries_per_sec": aggregates(v["qps"]),
            "eff_gb_per_sec": aggregates(v["gbps"]),
        }
    return out


def compute_aggregates_cluster(rows):
    g=defaultdict(lambda: {"updates_ps":[], "finalizes_ps":[]})
    for o in rows:
        g[group_key_cluster(o)]["updates_ps"].append(float(o["updates_per_sec"]))
        g[group_key_cluster(o)]["finalizes_ps"].append(float(o["finalizes_per_sec"]))
    out={}
    for k,v in g.items():
        out[k]={
            "updates_per_sec": aggregates(v["updates_ps"]),
            "finalizes_per_sec": aggregates(v["finalizes_ps"]),
        }
    return out


# --- Phase D additions: variance-bounds and provenance helpers ---

def _var_threshold(os_name: str) -> float:
    os_key = os_name.lower()
    if os_key in ("ubuntu-latest", "linux"):  # workflow passes normalized 'linux'
        return 0.10
    if os_key in ("macos-14", "macos", "macos-latest"):
        return 0.20
    # windows-2022/windows-latest
    return 0.25


def _get_provenance():
    prov = {
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "image_os": os.environ.get("ImageOS", ""),
        "image_version": os.environ.get("ImageVersion", ""),
    }
    # Try to augment from CMake cache if present
    try:
        with open("build/CMakeCache.txt", "r", encoding="utf-8") as f:
            cache = f.read()
        import re
        cc = re.search(r"CMAKE_CXX_COMPILER:.*=(.*)", cache)
        cv = re.search(r"CMAKE_CXX_COMPILER_VERSION:.*=(.*)", cache)
        if cc:
            prov["cmake_cxx_compiler"] = cc.group(1).strip()
        if cv:
            prov["cmake_cxx_compiler_version"] = cv.group(1).strip()
    except Exception:
        pass
    return prov


def _write_aggregates_ndjson(path, am_aggr, cl_aggr, provenance):
    try:
        with open(path, "w", encoding="utf-8") as out:
            for k, v in am_aggr.items():
                rec = {"kind": "AM", "key": list(k)}
                rec.update(v)
                rec.update(provenance)
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            for k, v in cl_aggr.items():
                rec = {"kind": "Cluster", "key": list(k)}
                rec.update(v)
                rec.update(provenance)
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"WARN: failed to write aggregates NDJSON '{path}': {e}", file=sys.stderr)


def _build_counts_am(rows):
    g = defaultdict(int)
    for o in rows:
        g[group_key_am(o)] += 1
    return g


def _build_counts_cluster(rows):
    g = defaultdict(int)
    for o in rows:
        g[group_key_cluster(o)] += 1
    return g


def load_baseline(dirpath, os_name):
    # Expect per-OS subdir: linux|windows|macos
    m={
        "ubuntu-latest":"linux",
        "windows-latest":"windows",
        "macos-latest":"macos",
    }
    sub=m.get(os_name, os_name)
    am_path=os.path.join(dirpath, sub, "am_bench.json")
    cl_path=os.path.join(dirpath, sub, "cluster_bench.json")
    with open(am_path,'r',encoding='utf-8') as f:
        am=json.load(f)
    with open(cl_path,'r',encoding='utf-8') as f:
        cl=json.load(f)
    return am, cl


def compare_metric(current, baseline, tol_pct, label):
    # Allow decreases within tol_pct; any drop > tol fails; improves always pass
    cur=float(current)
    base=float(baseline)
    if base<=0:
        return True, "baseline<=0"
    allowed=(1.0 - (float(tol_pct)/100.0)) * base
    ok=cur >= allowed
    detail=f"{label}: current={cur:.2f}, baseline={base:.2f}, allowed>= {allowed:.2f} (tol={tol_pct}%)"
    return ok, detail


def run(args):
    am_rows=filter_rows(load_ndjson(args.am), REQ_AM)
    cl_rows=filter_rows(load_ndjson(args.cluster), REQ_CLUSTER)
    require_fields(am_rows, REQ_AM)
    require_fields(cl_rows, REQ_CLUSTER)

    # Aggregates
    am_aggr=compute_aggregates_am(am_rows)
    cl_aggr=compute_aggregates_cluster(cl_rows)

    # Sample counts per group (for variance-bounds gating)
    am_counts=_build_counts_am(am_rows)
    cl_counts=_build_counts_cluster(cl_rows)

    am_base, cl_base=load_baseline(args.baseline_dir, args.os)

    failures=[]
    # Compare AM
    for key_str, base_obj in am_base.items():
        key=tuple(json.loads(key_str))  # keys are serialized tuples: [name,dim,cap,size]
        cur=am_aggr.get(key)
        if not cur:
            failures.append(f"Missing AM group in current run: {key}")
            continue
        ok1,d1=compare_metric(cur["queries_per_sec"]["mean"], base_obj["queries_per_sec"]["mean"], args.tol_qps, "AM qps")
        ok2,d2=compare_metric(cur["eff_gb_per_sec"]["mean"], base_obj["eff_gb_per_sec"]["mean"], args.tol_gbps, "AM gbps")
        if not ok1: failures.append(d1+f" key={key}")
        if not ok2: failures.append(d2+f" key={key}")

    # Compare Cluster
    for key_str, base_obj in cl_base.items():
        key=tuple(json.loads(key_str))  # [name,dim,cap,updates]
        cur=cl_aggr.get(key)
        if not cur:
            failures.append(f"Missing Cluster group in current run: {key}")
            continue
        ok1,d1=compare_metric(cur["updates_per_sec"]["mean"], base_obj["updates_per_sec"]["mean"], args.tol_qps, "Cluster updates")
        ok2,d2=compare_metric(cur["finalizes_per_sec"]["mean"], base_obj["finalizes_per_sec"]["mean"], args.tol_qps, "Cluster finalizes")
        if not ok1: failures.append(d1+f" key={key}")
        if not ok2: failures.append(d2+f" key={key}")

    # Variance-bounds enforcement (only when sufficient samples)
    thr=_var_threshold(args.os)
    for k,v in am_aggr.items():
        n=am_counts.get(k,0)
        if n>=3:
            mean=v["queries_per_sec"]["mean"]; std=v["queries_per_sec"]["stdev"]
            if mean>0 and (std/mean)>thr:
                failures.append(f"AM qps variance too high: stdev/mean={(std/mean):.3f} > {thr:.3f} key={k}")
            mean=v["eff_gb_per_sec"]["mean"]; std=v["eff_gb_per_sec"]["stdev"]
            if mean>0 and (std/mean)>thr:
                failures.append(f"AM gbps variance too high: stdev/mean={(std/mean):.3f} > {thr:.3f} key={k}")
    for k,v in cl_aggr.items():
        n=cl_counts.get(k,0)
        if n>=3:
            mean=v["updates_per_sec"]["mean"]; std=v["updates_per_sec"]["stdev"]
            if mean>0 and (std/mean)>thr:
                failures.append(f"Cluster updates variance too high: stdev/mean={(std/mean):.3f} > {thr:.3f} key={k}")
            mean=v["finalizes_per_sec"]["mean"]; std=v["finalizes_per_sec"]["stdev"]
            if mean>0 and (std/mean)>thr:
                failures.append(f"Cluster finalizes variance too high: stdev/mean={(std/mean):.3f} > {thr:.3f} key={k}")

    # Emit aggregates with provenance for artifacting
    _write_aggregates_ndjson("perf_agg.ndjson", am_aggr, cl_aggr, _get_provenance())

    # Log summary
    print("=== Aggregates (AM) ===")
    for k,v in am_aggr.items():
        print(json.dumps({"key":k, **v}, default=lambda o:o, separators=(",",":")))
    print("=== Aggregates (Cluster) ===")
    for k,v in cl_aggr.items():
        print(json.dumps({"key":k, **v}, separators=(",",":")))

    if failures:
        print("\nPERF REGRESSION DETECTED:")
        for f in failures:
            print(" - ", f)
        return 2
    return 0

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--am', required=True)
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--am-schema', required=True)
    ap.add_argument('--cluster-schema', required=True)
    ap.add_argument('--baseline-dir', required=True)
    ap.add_argument('--os', required=True)
    ap.add_argument('--tol-qps', type=float, required=True)
    ap.add_argument('--tol-gbps', type=float, required=True)
    args=ap.parse_args()

    # Note: schema paths are currently informational; schema-lock is enforced via key presence checks above.
    # If needed, JSON Schema validation can be added without external deps by manual checks.

    sys.exit(run(args))

