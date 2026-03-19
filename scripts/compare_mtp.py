#!/usr/bin/env python3
"""Compare MTP0 vs MTP3 benchmark results side-by-side.

Shows throughput gain, latency reduction, and MTP acceptance rate.

Usage:
    python scripts/compare_mtp.py runs/8xb200-fp8-*-mtp0.json runs/8xb200-fp8-*-mtp3.json
    python scripts/compare_mtp.py --mtp0 runs/mtp0.json --mtp3 runs/mtp3.json
"""

import argparse
import json
import sys


def load_run(path):
    with open(path) as f:
        return json.load(f)


def build_index(run):
    """Index results by (scenario, conc) for O(1) lookup."""
    idx = {}
    for r in run["results"]:
        key = (r["scenario"], r["conc"])
        idx[key] = r
    return idx


def estimate_acceptance_rate(mtp0, mtp3, mtp_layers=3):
    """Estimate MTP acceptance rate from TPOT ratio.

    MTP speculative decoding proposes `mtp_layers` extra tokens per step.
    If all accepted: speedup = (1 + mtp_layers)x on TPOT.
    acceptance_rate ≈ (speedup - 1) / mtp_layers
    where speedup = tpot_mtp0 / tpot_mtp3
    """
    tpot0 = mtp0.get("tpot_p50", 0)
    tpot3 = mtp3.get("tpot_p50", 0)
    if not tpot3 or not tpot0:
        return None
    speedup = tpot0 / tpot3
    # Clamp: speedup can't exceed (1 + mtp_layers) theoretically
    acceptance = (speedup - 1) / mtp_layers
    return min(max(acceptance, 0), 1.0)


def main():
    parser = argparse.ArgumentParser(description="Compare MTP0 vs MTP3 results")
    parser.add_argument("--mtp0", required=True, help="MTP0 run JSON")
    parser.add_argument("--mtp3", required=True, help="MTP3 run JSON")
    parser.add_argument("--mtp-layers", type=int, default=3, help="Number of MTP layers (default: 3)")
    args = parser.parse_args()

    run0 = load_run(args.mtp0)
    run3 = load_run(args.mtp3)
    idx0 = build_index(run0)
    idx3 = build_index(run3)

    # Header
    print(f"MTP0: {args.mtp0}")
    print(f"MTP3: {args.mtp3}")
    print(f"MTP layers: {args.mtp_layers}")
    print()

    hdr = (
        f"{'Scenario':<12} {'CONC':>5} │ "
        f"{'Out TPS':>9} {'Out TPS':>9} {'Δ%':>7} │ "
        f"{'TPOT p50':>9} {'TPOT p50':>9} {'Δ%':>7} │ "
        f"{'TTFT p50':>9} {'TTFT p50':>9} {'Δ%':>7} │ "
        f"{'Accept':>7}"
    )
    sub = (
        f"{'':12} {'':>5} │ "
        f"{'mtp0':>9} {'mtp3':>9} {'':>7} │ "
        f"{'mtp0':>9} {'mtp3':>9} {'':>7} │ "
        f"{'mtp0':>9} {'mtp3':>9} {'':>7} │ "
        f"{'rate':>7}"
    )
    sep = "─" * len(hdr)

    print(sub)
    print(hdr)
    print(sep)

    # All keys, sorted by scenario then conc
    all_keys = sorted(set(list(idx0.keys()) + list(idx3.keys())),
                      key=lambda k: ({"chat": 0, "reasoning": 1, "summarize": 2}.get(k[0], 9), k[1]))

    prev_scenario = None
    for scenario, conc in all_keys:
        r0 = idx0.get((scenario, conc))
        r3 = idx3.get((scenario, conc))

        if not r0 or not r3:
            continue

        if prev_scenario and prev_scenario != scenario:
            print(sep)
        prev_scenario = scenario

        out0 = r0["output_tps"]
        out3 = r3["output_tps"]
        out_delta = (out3 - out0) / out0 * 100 if out0 else 0

        tpot0 = r0["tpot_p50"]
        tpot3 = r3["tpot_p50"]
        tpot_delta = (tpot3 - tpot0) / tpot0 * 100 if tpot0 else 0

        ttft0 = r0["ttft_p50"]
        ttft3 = r3["ttft_p50"]
        ttft_delta = (ttft3 - ttft0) / ttft0 * 100 if ttft0 else 0

        acc = estimate_acceptance_rate(r0, r3, args.mtp_layers)
        acc_str = f"{acc:.1%}" if acc is not None else "N/A"

        print(
            f"{scenario:<12} {conc:>5} │ "
            f"{out0:>9.1f} {out3:>9.1f} {out_delta:>+6.1f}% │ "
            f"{tpot0:>8.2f}ms {tpot3:>8.2f}ms {tpot_delta:>+6.1f}% │ "
            f"{ttft0:>8.1f}ms {ttft3:>8.1f}ms {ttft_delta:>+6.1f}% │ "
            f"{acc_str:>7}"
        )

    print(sep)
    print()
    print("Δ%: positive = MTP3 higher (better for TPS, worse for latency)")
    print("Accept rate: estimated from TPOT speedup = (tpot_mtp0/tpot_mtp3 - 1) / mtp_layers")


if __name__ == "__main__":
    main()
