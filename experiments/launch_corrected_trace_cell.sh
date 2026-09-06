#!/usr/bin/env bash
set -u

if (( $# != 5 )); then
  printf 'usage: %s FAMILY METHOD CLIP SEED PHYSICAL_GPU\n' "$0" >&2
  exit 64
fi
family=$1
method=$2
clip=$3
seed=$4
gpu=$5
case "$method" in gd|rank1_gd|diag_gd) ;; *) printf 'invalid method\n' >&2; exit 64 ;; esac
case "$seed" in 3407|3408|3409) ;; *) printf 'invalid seed\n' >&2; exit 64 ;; esac
case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) printf 'invalid GPU\n' >&2; exit 64 ;; esac
case "$family:$clip" in
  r23:1|r23:1000000)
    run_root=runs/r23_corrected_trace
    checkpoint=../checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors
    stem="${method}_clip${clip}_s${seed}"
    ;;
  r24:1)
    run_root=runs/r24_scale1028_corrected_trace
    checkpoint=../checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors
    stem="${method}_s${seed}"
    ;;
  *) printf 'invalid family/clip pair\n' >&2; exit 64 ;;
esac

cd /home/JJ_Group/lih2511/test/dllm/iclr_2 || exit 72
output="$run_root/formal/${stem}.json"
log="$run_root/logs/${stem}.log"
exit_file="$run_root/logs/${stem}.exit"
samples="$run_root/logs/${stem}_gpu_memory.csv"
peak="$run_root/logs/${stem}_peak_gpu_mib.txt"
for target in "$output" "$log" "$exit_file" "$samples" "$peak"; do
  test ! -e "$target" || { printf 'refusing existing target: %s\n' "$target" >&2; exit 73; }
done

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu"
printf 'utc,memory_used_mib\n' >"$samples"
{
  printf 'launch_utc=%s host=%s physical_gpu=%s family=%s method=%s clip=%s seed=%s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$gpu" "$family" "$method" "$clip" "$seed"
  sha256sum experiments/dllm_corrected_trace_control.py \
    experiments/summarize_corrected_trace_controls.py "$run_root/contract.json"
  /usr/bin/time -v timeout --signal=TERM --kill-after=60s 7200 \
    /home/JJ_Group/lih2511/.conda/envs/smdm-baseline/bin/python \
    experiments/dllm_corrected_trace_control.py \
    --family "$family" --method "$method" --b-clip "$clip" --seed "$seed" \
    --checkpoint "$checkpoint" --tokenizer tokenizer \
    --reverse-dir SMDM/data/reverse_experiments/june_version_7921032488 \
    --device cuda --output "$output"
} >"$log" 2>&1 &
job=$!
while kill -0 "$job" 2>/dev/null; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%s\n' "$(date -u +%FT%TZ)" "$used" >>"$samples"
  sleep 2
done
wait "$job"
code=$?
printf '%s\n' "$code" >"$exit_file"
awk -F, 'NR > 1 && $2 + 0 > peak { peak=$2 + 0 } END { print peak + 0 }' "$samples" >"$peak"
exit "$code"
