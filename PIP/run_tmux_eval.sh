#!/bin/bash

models=(
    xxx
    xxx
    xxx
)


for model in "${models[@]}"; do
    session_name="eval_domain_${model}"
    tmux new-session -d -s "${session_name}"

    tmux send-keys -t "${session_name}:" "
        source xxx/miniconda/bin/activate base 

        python /public_api_domain_eval.py \
            --batch 500 \
            --model ${model} > logs/domain_eval_${model}.log 2>&1
        exit
    " Enter
done


for model in "${models[@]}"; do
    session_name="eval_onlyOne_${model}"
    tmux new-session -d -s "${session_name}"

    tmux send-keys -t "${session_name}:" "
        source xxx/miniconda/bin/activate base 

        python public_api_bench_eval.py \
            --batch 500 \
            --model ${model} > logs/onlyOne_eval_${model}.log 2>&1

        exit
    " Enter
done