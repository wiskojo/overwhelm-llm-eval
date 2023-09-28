#!/bin/bash

SEED=0

python experiment.py --num_trials 20 --eval_model gpt-3.5-turbo --requests ./data/requests_gpt_35_turbo.jsonl --results ./data/results_gpt_35_turbo.jsonl --seed $SEED --skip_confirmation
python experiment.py --num_trials 20 --eval_model meta-llama/Llama-2-70b-chat-hf --requests ./data/requests_llama_2_70b_chat.jsonl --results ./data/results_llama_2_70b_chat.jsonl --seed $SEED --skip_confirmation
python experiment.py --num_trials 5 --eval_model gpt-4 --requests ./data/requests_gpt_4.jsonl --results ./data/results_gpt_4.jsonl --seed $SEED --skip_confirmation
