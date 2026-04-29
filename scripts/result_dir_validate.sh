# Shared result_dir naming convention validator.
# Sourced by ci_lib.sh (host) and bench/collect scripts (container).
#
# Required pattern:
#   results/<platform>_<model>_<quant>/<platform>_<model>_<quant>_mtp<N>_ep<N>_tp<N>_<env>[_<suffix>]
#
# Example compliant:
#   results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep8_tp8_post3
#   results/b300_gptoss120b_mxfp4/b300_gptoss120b_mxfp4_mtp0_ep1_tp1_post3_profiling

validate_result_dir() {
  local d=$1
  if [[ "$d" =~ ^\.?/?results/[a-z0-9]+_[a-z0-9]+_[a-z0-9]+/[a-z0-9]+_[a-z0-9]+_[a-z0-9]+_mtp[0-9]+_ep[0-9]+_tp[0-9]+_[a-z0-9_]+/?$ ]]; then
    return 0
  fi
  echo "ERROR: --result-dir '$d' violates naming convention." >&2
  echo "  Required: results/<platform>_<model>_<quant>/<platform>_<model>_<quant>_mtp<N>_ep<N>_tp<N>_<env>[_<suffix>]" >&2
  echo "  Example:  results/b200_dsr_fp4/b200_dsr_fp4_mtp0_ep8_tp8_post3" >&2
  echo "  Use ci_result_dir in scripts/ci_lib.sh to generate compliant names." >&2
  return 1
}
