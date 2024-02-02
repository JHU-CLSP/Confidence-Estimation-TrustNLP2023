# T0 - majority
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/T0/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/T0/cos_e' \
  --paraphrase_input_dir '../../data/predictions/paraphrase/T0/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name T0 \
  --probability_col log_probabilities \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority majority

## FLAN - majority
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/flan/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/flan/cos_e' \
  --paraphrase_input_dir '../../data/predictions/paraphrase/flan/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name flan \
  --probability_col log_probabilities \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority majority

## GPT - majority
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/gpt/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/gpt/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name gpt \
  --probability_col total_logprob \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority majority

# T0 - rand
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/T0/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/T0/cos_e' \
  --paraphrase_input_dir '../../data/predictions/paraphrase/T0/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name T0 \
  --probability_col log_probabilities \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority rand

# FLAN - rand
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/flan/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/flan/cos_e' \
  --paraphrase_input_dir '../../data/predictions/paraphrase/flan/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name flan \
  --probability_col log_probabilities \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority rand

# GPT - rand
python compute_ece.py \
  --OIP_input_dir '../../data/predictions/OIP_prompt/gpt/cos_e' \
  --T0_prompt_input_dir '../../data/predictions/T0_prompts/gpt/cos_e' \
  --output_dir '../../data/statistics/ece' \
  --model_name gpt \
  --probability_col total_logprob \
  --sorting_algorithm mergesort \
  --rank_by_rand_or_majority rand
