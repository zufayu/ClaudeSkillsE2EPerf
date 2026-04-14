#!/usr/bin/env python3
"""Generate decode_walltime CSV from torch trace .json.gz files.

Usage: python3 gen_decode_walltime.py <source_dir> <target_dir>

Scans source_dir for .pt.trace.json.gz files, extracts decode events,
and writes decode_walltime_trace_chat_c4_tp8_p40.csv to target_dir.
"""
import gzip, json, csv, sys, os, glob

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <source_dir> <target_dir>")
        sys.exit(1)

    src_dir = sys.argv[1]
    tgt_dir = sys.argv[2]

    # Find trace file
    traces = sorted(glob.glob(os.path.join(src_dir, "*.pt.trace.json.gz")))
    traces = [t for t in traces if "capture_graph" not in t and "trim" not in t]
    if not traces:
        print(f"No trace files found in {src_dir}")
        sys.exit(1)

    trace_path = traces[0]
    print(f"Loading {trace_path}...")
    with gzip.open(trace_path, "rt") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    print(f"Total events: {len(events)}")

    decodes = []
    for e in events:
        if (e.get("ph") == "X"
            and "user_annotation" in e.get("cat", "")
            and (e.get("name", "").startswith("decode[")
                 or e.get("name", "").startswith("decode "))):
            name = e["name"]
            bs = "unknown"
            for part in name.replace("[", " ").replace("]", " ").replace(",", " ").split():
                if part.startswith("bs=") or part.startswith("bs:"):
                    bs = part.split("=")[-1].split(":")[-1]
            decodes.append({"ts": e["ts"], "dur": e["dur"], "bs": bs, "name": name})

    decodes.sort(key=lambda x: x["ts"])
    print(f"Decode events: {len(decodes)}")

    os.makedirs(tgt_dir, exist_ok=True)
    out_csv = os.path.join(tgt_dir, "decode_walltime_trace_chat_c4_tp8_p40.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "ts_us", "dur_us", "bs"])
        for i, d in enumerate(decodes):
            w.writerow([i, d["ts"], d["dur"], d["bs"]])

    print(f"Wrote {len(decodes)} rows to {out_csv}")

if __name__ == "__main__":
    main()
