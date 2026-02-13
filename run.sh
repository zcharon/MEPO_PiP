ray job submit --address="address" \
    --working-dir=. \
    --no-wait \
    -- \
    python3 -m verl.trainer.main \
    config=recipes/mgdo_qwen3vl_30ba3b_batch128_lf.yaml 