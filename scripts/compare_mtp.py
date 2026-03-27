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
        total0, total3 = r0.get("total_tps", 0), r3.get("total_tps", 0)
        out0, out3 = r0["output_tps"], r3["output_tps"]
        tpot0, tpot3 = r0["tpot_p50"], r3["tpot_p50"]
        ttft0, ttft3 = r0["ttft_p50"], r3["ttft_p50"]
        inter0 = 1000.0 / tpot0 if tpot0 > 0 else 0
        inter3 = 1000.0 / tpot3 if tpot3 > 0 else 0
        total_delta = (total3 - total0) / total0 * 100 if total0 else 0
        out_delta = (out3 - out0) / out0 * 100 if out0 else 0
        tpot_delta = (tpot3 - tpot0) / tpot0 * 100 if tpot0 else 0
        ttft_delta = (ttft3 - ttft0) / ttft0 * 100 if ttft0 else 0
        inter_delta = (inter3 - inter0) / inter0 * 100 if inter0 else 0
        rows.append({
            "scenario": scenario, "conc": conc,
            "total0": total0, "total3": total3, "total_delta": total_delta,
            "out0": out0, "out3": out3, "out_delta": out_delta,
            "tpot0": tpot0, "tpot3": tpot3, "tpot_delta": tpot_delta,
            "ttft0": ttft0, "ttft3": ttft3, "ttft_delta": ttft_delta,
            "inter0": inter0, "inter3": inter3, "inter_delta": inter_delta,
        })
    return rows


def print_terminal(rows, label0, label3):
    """Print comparison as terminal table."""
    hdr = (
        f"{'Scenario':<12} {'CONC':>5} | "
        f"{'Total':>9} {'Total':>9} {'D%':>7} | "
        f"{'Out TPS':>9} {'Out TPS':>9} {'D%':>7} | "
        f"{'Interac':>9} {'Interac':>9} {'D%':>7} | "
        f"{'TPOT p50':>9} {'TPOT p50':>9} {'D%':>7} | "
        f"{'TTFT p50':>9} {'TTFT p50':>9} {'D%':>7}"
    )
    sub = (
        f"{'':12} {'':>5} | "
        f"{label0:>9} {label3:>9} {'':>7} | "
        f"{label0:>9} {label3:>9} {'':>7} | "
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
            f"{r['total0']:>9.1f} {r['total3']:>9.1f} {r['total_delta']:>+6.1f}% | "
            f"{r['out0']:>9.1f} {r['out3']:>9.1f} {r['out_delta']:>+6.1f}% | "
            f"{r['inter0']:>9.1f} {r['inter3']:>9.1f} {r['inter_delta']:>+6.1f}% | "
            f"{r['tpot0']:>8.2f}ms {r['tpot3']:>8.2f}ms {r['tpot_delta']:>+6.1f}% | "
            f"{r['ttft0']:>8.1f}ms {r['ttft3']:>8.1f}ms {r['ttft_delta']:>+6.1f}%"
        )
    print(sep)


def print_md_single(rows, label0, label3):
    """Print comparison as markdown table."""
    print(f"| Scenario | Conc | Total {label0} | Total {label3} | Total Gain | "
          f"Out {label0} | Out {label3} | Out Gain | "
          f"Interac {label0} | Interac {label3} | Interac Gain | "
          f"TPOT {label0} | TPOT {label3} | TPOT Chg | "
          f"TTFT {label0} | TTFT {label3} | TTFT Chg |")
    print("|----------|------|" + "-----------|" * 15)
    prev = None
    for r in rows:
        if prev and prev != r["scenario"]:
            print(f"| | |" + " |" * 15)
        prev = r["scenario"]
        print(
            f"| {r['scenario']} | {r['conc']} | "
            f"{r['total0']:.1f} | {r['total3']:.1f} | {r['total_delta']:+.1f}% | "
            f"{r['out0']:.1f} | {r['out3']:.1f} | {r['out_delta']:+.1f}% | "
            f"{r['inter0']:.1f} | {r['inter3']:.1f} | {r['inter_delta']:+.1f}% | "
            f"{r['tpot0']:.2f} | {r['tpot3']:.2f} | {r['tpot_delta']:+.1f}% | "
            f"{r['ttft0']:.1f} | {r['ttft3']:.1f} | {r['ttft_delta']:+.1f}% |"
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

    metrics_def = [
        ("Total Tput", "total", "total_delta"),
        ("Output Tput", "out", "out_delta"),
        ("Interactivity", "inter", "inter_delta"),
        ("TPOT", "tpot", "tpot_delta"),
        ("TTFT", "ttft", "ttft_delta"),
    ]

    for metric_name, key_prefix, delta_key in metrics_def:
        print(f"\n## {metric_name}\n")
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

                k0, k3 = f"{key_prefix}0", f"{key_prefix}3"
                fmt = ".2f" if key_prefix == "tpot" else ".1f"
                b_str0 = f"{b[k0]:{fmt}}" if b else "-"
                b_str3 = f"{b[k3]:{fmt}}" if b else "-"
                b_gain = f"{b[delta_key]:+.1f}%" if b else "-"
                m_str0 = f"{m[k0]:{fmt}}" if m else "-"
                m_str3 = f"{m[k3]:{fmt}}" if m else "-"
                m_gain = f"{m[delta_key]:+.1f}%" if m else "-"

                if b and m:
                    # For TPOT/TTFT lower is better, so bigger negative delta = better
                    if key_prefix in ("tpot", "ttft"):
                        winner = "B200" if b[delta_key] < m[delta_key] else "355X"
                    else:
                        winner = "B200" if b[delta_key] > m[delta_key] else "355X"
                elif b:
                    winner = "B200 only"
                else:
                    winner = "355X only"

                print(f"| {conc} | {b_str0} | {b_str3} | {b_gain} | "
                      f"{m_str0} | {m_str3} | {m_gain} | {winner} |")

    # Summary scoreboard across all 5 metrics
    print(f"\n## Scoreboard\n")
    print(f"| Metric | B200 Wins | 355X Wins | Total |")
    print(f"|--------|-----------|-----------|-------|")
    common_keys = set(b200_by.keys()) & set(mi_by.keys())
    grand_b200 = 0
    grand_mi = 0
    for metric_name, key_prefix, delta_key in metrics_def:
        b_wins = 0
        m_wins = 0
        for key in common_keys:
            if key_prefix in ("tpot", "ttft"):
                if b200_by[key][delta_key] < mi_by[key][delta_key]:
                    b_wins += 1
                else:
                    m_wins += 1
            else:
                if b200_by[key][delta_key] > mi_by[key][delta_key]:
                    b_wins += 1
                else:
                    m_wins += 1
        grand_b200 += b_wins
        grand_mi += m_wins
        print(f"| {metric_name} | {b_wins} | {m_wins} | {b_wins + m_wins} |")
    print(f"| **Grand Total** | **{grand_b200}** | **{grand_mi}** | **{grand_b200 + grand_mi}** |")


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
