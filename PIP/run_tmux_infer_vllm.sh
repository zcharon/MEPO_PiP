#!/bin/bash

models=(
    xxx
    xxx
    xxx
)

benchmarks=("blink" "mathverse" "mmmu_pro" "mmstar" "muirbench" "realworldqa" "remi")
n=1
env_path="xxx"
session_name="1"
tmp_script="/tmp/run_all_infer_serial.sh"
echo "#!/bin/bash" > $tmp_script
echo "source $env_path" >> $tmp_script
echo "" >> $tmp_script
for model in "${models[@]}"; do
    for benchmark in "${benchmarks[@]}"; do
        echo "echo '==== Running model: $model | benchmark: $benchmark ===='" >> $tmp_script
        echo "python3 -u vllm_domain_infer.py \\" >> $tmp_script
        echo "  --model $model --n $n --batch 256 --benchmark $benchmark \\" >> $tmp_script
        echo "  >logs/domain_infer_${model}_${benchmark}.log 2>&1" >> $tmp_script
        echo "echo '==== Finished $model | $benchmark ===='" >> $tmp_script
        echo "" >> $tmp_script
    done
done
echo "exit" >> $tmp_script
chmod +x $tmp_script
tmux new-session -d -s "$session_name" "bash $tmp_script"

n=1  
env_path="env_path"
session_name="1"
tmux new-session -d -s "$session_name"
cmd=""
for model in "${models[@]}"; do
    cmd+="source $env_path && "
    cmd+="python3 -u vllm_infer.py --model $model --n $n --batch 256 "
    cmd+=">logs/onlyBench_infer_${model}.log 2>&1 && "
done
cmd=${cmd% && }
cmd+="; exit"
tmux send-keys -t "$session_name" "$cmd" Enter