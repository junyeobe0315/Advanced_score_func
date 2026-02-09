#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs}"
NFE_LIST="${NFE_LIST:-10,20,50,100,200}"
OUT_DIR="${OUT_DIR:-reports}"
STEPS_SAMPLER="${STEPS_SAMPLER:-heun}"
COMPUTE_SAMPLER="${COMPUTE_SAMPLER:-heun}"

mapfile -t run_dirs < <(find "$RUN_ROOT" -mindepth 3 -maxdepth 3 -type d -name "seed*" | sort -V)

if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "[warn] no run directories found under ${RUN_ROOT}"
fi

ok_count=0
fail_count=0
skip_count=0

for run_dir in "${run_dirs[@]}"; do
  if [[ ! -f "${run_dir}/config_resolved.yaml" ]]; then
    echo "[skip] missing config_resolved.yaml: ${run_dir}"
    ((skip_count += 1))
    continue
  fi

  if ! compgen -G "${run_dir}/checkpoints/step_*.pt" > /dev/null; then
    echo "[skip] missing checkpoints: ${run_dir}"
    ((skip_count += 1))
    continue
  fi

  echo "[eval] ${run_dir}"
  if python -m src.main_eval --run_dir "${run_dir}" --nfe_list "${NFE_LIST}"; then
    ((ok_count += 1))
  else
    echo "[fail] evaluation failed: ${run_dir}"
    ((fail_count += 1))
  fi
done

echo "[report] aggregate tables -> ${OUT_DIR}"
python scripts/make_report_tables.py \
  --run_root "${RUN_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --steps_sampler "${STEPS_SAMPLER}" \
  --compute_sampler "${COMPUTE_SAMPLER}"

echo "[done] eval_ok=${ok_count} eval_fail=${fail_count} skipped=${skip_count} out_dir=${OUT_DIR}"
