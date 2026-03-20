#!/usr/bin/env python3
"""Compare MTP0 vs MTP3 benchmark results side-by-side.

Shows throughput gain and latency reduction.

Usage:
    # Terminal table:
    python scripts/compare_mtp.py --mtp0 runs/mtp0.json --mtp3 runs/mtp3.json

    # Markdown table (for pasting into reports):
    python scripts/compare_mtp.py --mtp0 runs/mtp0.json --mtp3 runs/mtp3.json --md

    # Cross-platform comparison (generates full report tables):
    python scripts/compare_mtp.py --cross \
        --b200-mtp0 runs/8xb200-fp8-*-mtp0*.json \
        --b200-mtp3 runs/8xb200-fp8-*-mtp3*.json \
        --mi355x-mtp0 runs/atom-mi355x-*.json \
        --mi355x-mtp3 runs/atom-mi355x-*-mtp3.json \
        --md
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


SCENARIO_ORDER = {"chat": 0, "reasoning": 1, "summarize": 2}
SCENARIO_ISL_OSL = {"chat": "1K/1K", "reasoning": "1K/8K", "summarize": "8K/1K"}


def compute_pairs(idx0, idx3):
    """Compute comparison rows for matched (scenario, conc) pairs."""
    all_keys = sorted(
        set(idx0.keys()) & set(idx3.keys()),
        key=lambda k: (SCENARIO_ORDER.get(k[0], 9), k[1]),
    )
    rows = []
    for scenario, conc in all_keys:
        r0, r3 = idx0[(scenario, conc)], idx3[(scenario, conc)]
        out0, out3 = r0["output_tps"], r3["output_tps"]
        tpot0, tpot3 = r0["tpot_p50"], r3["tpot_p50"]
        ttft0, ttft3 = r0["ttft_p50"], r3["ttft_p50"]
        out_delta = (out3 - out0) / out0 * 100 if out0 else 0
        tpot_delta = (tpot3 - tpot0) / tpot0 * 100 if tpot0 else 0
        ttft_delta = (ttft3 - ttft0) / ttft0 * 100 if ttft0 else 0
        rows.append({
            "scenario": scenario, "conc": conc,
            "out0": out0, "out3": out3, "out_delta": out_delta,
            "tpot0": tpot0, "tpot3": tpot3, "tpot_delta": tpot_delta,
            "ttft0": ttft0, "ttft3": ttft3, "ttft_delta": ttft_delta,
        })
    return rows


def print_terminal(rows, label0, label3):
    """Print comparison as terminal table."""
    hdr = (
        f"{'Scenario':<12} {'CONC':>5} | "
        f"{'Out TPS':>9} {'Out TPS':>9} {'D%':>7} | "
        f"{'TPOT p50':>9} {'TPOT p50':>9} {'D%':>7} | "
        f"{'TTFT p50':>9} {'TTFT p50':>9} {'D%':>7}"
    )
    sub = (
        f"{'':12} {'':>5} | "
        f"{label0:>9} {label3:>9} {'':>7} | "
        f"{label0:>9} {label3:>9} {'':>7} | "
        f"{label0:>9} {label3:>9} {'':>7}"
    )
    sep = "-" * len(hdr)
    print(sub)
    print(hdr)
    print(sep)
    prev = None
    for r in rows:
        if prev and prev != r["scenario"]:
            print(sep)
        prev = r["scenario"]
        print(
            f"{r['scenario']:<12} {r['conc']:>5} | "
            f"{r['out0']:>9.1f} {r['out3']:>9.1f} {r['out_delta']:>+6.1f}% | "
            f"{r['tpot0']:>8.2f}ms {r['tpot3']:>8.2f}ms {r['tpot_delta']:>+6.1f}% | "
            f"{r['ttft0']:>8.1f}ms {r['ttft3']:>8.1f}ms {r['ttft_delta']:>+6.1f}%"
        )
    print(sep)


def print_md_single(rows, label0, label3):
    """Print comparison as markdown table."""
    print(f"| Scenario | Conc | {label0} TPS | {label3} TPS | TPS Gain | "
          f"TPOT {label0} | TPOT {label3} | TPOT Chg |")
    print("|----------|------|-----------|-----------|----------|"
          "----------|----------|----------|")
    prev = None
    for r in rows:
        if prev and prev != r["scenario"]:
            print(f"| | | | | | | | |")
        prev = r["scenario"]
        print(
            f"| {r['scenario']} | {r['conc']} | "
            f"{r['out0']:.1f} | {r['out3']:.1f} | {r['out_delta']:+.1f}% | "
            f"{r['tpot0']:.2f}ms | {r['tpot3']:.2f}ms | {r['tpot_delta']:+.1f}% |"
        )


def print_md_cross(b200_rows, mi355x_rows):
    """Print cross-platform comparison as markdown tables, one per scenario."""
    # Group by scenario
    scenarios = []
    seen = set()
    for r in b200_rows + mi355x_rows:
        if r["scenario"] not in seen:
            scenarios.append(r["scenario"])
            seen.add(r["scenario"])
    scenarios.sort(key=lambda s: SCENARIO_ORDER.get(s, 9))

    b200_by = {}
    for r in b200_rows:
        b200_by[(r["scenario"], r["conc"])] = r
    mi_by = {}
    for r in mi355x_rows:
        mi_by[(r["scenario"], r["conc"])] = r

    for scenario in scenarios:
        isl_osl = SCENARIO_ISL_OSL.get(scenario, "?/?")
        print(f"\n### {scenario.capitalize()} ({isl_osl})\n")
        print("| Conc | B200 mtp0 | B200 mtp3 | B200 Gain | "
              "355X mtp0 | 355X mtp3 | 355X Gain | Winner |")
        print("|------|-----------|-----------|-----------|"
              "-----------|-----------|-----------|--------|")

        concs = sorted(set(
            [k[1] for k in b200_by if k[0] == scenario] +
            [k[1] for k in mi_by if k[0] == scenario]
        ))

        for conc in concs:
            b = b200_by.get((scenario, conc))
            m = mi_by.get((scenario, conc))

            b_str0 = f"{b['out0']:.1f}" if b else "-"
            b_str3 = f"{b['out3']:.1f}" if b else "-"
            b_gain = f"{b['out_delta']:+.1f}%" if b else "-"
            m_str0 = f"{m['out0']:.1f}" if m else "-"
            m_str3 = f"{m['out3']:.1f}" if m else "-"
            m_gain = f"{m['out_delta']:+.1f}%" if m else "-"

            if b and m:
                winner = "B200" if b["out_delta"] > m["out_delta"] else "355X"
            elif b:
                winner = "B200 only"
            else:
                winner = "355X only"

            print(f"| {conc} | {b_str0} | {b_str3} | {b_gain} | "
                  f"{m_str0} | {m_str3} | {m_gain} | {winner} |")

    # Summary scoreboard
    b200_wins = 0
    mi_wins = 0
    for key in set(b200_by.keys()) & set(mi_by.keys()):
        if b200_by[key]["out_delta"] > mi_by[key]["out_delta"]:
            b200_wins += 1
        else:
            mi_wins += 1
    total = b200_wins + mi_wins
    print(f"\n### Scoreboard\n")
    print(f"| | B200 | 355X | Total |")
    print(f"|------|------|------|-------|")
    print(f"| TPS gain winner | {b200_wins} | {mi_wins} | {total} |")


def main():
    parser = argparse.ArgumentParser(description="Compare MTP0 vs MTP3 results")
    parser.add_argument("--mtp0", help="MTP0 run JSON")
    parser.add_argument("--mtp3", help="MTP3 run JSON")
    parser.add_argument("--md", action="store_true", help="Output as markdown table")
    parser.add_argument("--cross", action="store_true", help="Cross-platform comparison mode")
    parser.add_argument("--b200-mtp0", help="B200 MTP0 run JSON (cross mode)")
    parser.add_argument("--b200-mtp3", help="B200 MTP3 run JSON (cross mode)")
    parser.add_argument("--mi355x-mtp0", help="MI355X MTP0 run JSON (cross mode)")
    parser.add_argument("--mi355x-mtp3", help="MI355X MTP3 run JSON (cross mode)")
    args = parser.parse_args()

    if args.cross:
        if not all([args.b200_mtp0, args.b200_mtp3, args.mi355x_mtp0, args.mi355x_mtp3]):
            parser.error("--cross requires --b200-mtp0, --b200-mtp3, --mi355x-mtp0, --mi355x-mtp3")

        b200_rows = compute_pairs(
            build_index(load_run(args.b200_mtp0)),
            build_index(load_run(args.b200_mtp3)),
        )
        mi_rows = compute_pairs(
            build_index(load_run(args.mi355x_mtp0)),
            build_index(load_run(args.mi355x_mtp3)),
        )

        if args.md:
            print_md_cross(b200_rows, mi_rows)
        else:
            print("=== B200 ===")
            print_terminal(b200_rows, "mtp0", "mtp3")
            print("\n=== MI355X ===")
            print_terminal(mi_rows, "mtp0", "mtp3")
    else:
        if not args.mtp0 or not args.mtp3:
            parser.error("--mtp0 and --mtp3 are required (or use --cross mode)")

        run0 = load_run(args.mtp0)
        run3 = load_run(args.mtp3)
        rows = compute_pairs(build_index(run0), build_index(run3))

        label0 = run0.get("env_tag", "mtp0")
        label3 = run3.get("env_tag", "mtp3")

        if args.md:
            print_md_single(rows, label0, label3)
        else:
            print_terminal(rows, label0, label3)


if __name__ == "__main__":
    main()
