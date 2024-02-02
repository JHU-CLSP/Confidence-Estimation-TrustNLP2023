
# Strength in Numbers: Estimating Confidence of  Large Language Models by Prompt Agreement
Gwenyth Portillo Wightman and Alexandra DeLucia and Mark Dredze
  
This repository contains information for replicating the approach used in [Strength in Numbers: Estimating Confidence of  Large Language Models by Prompt Agreement](https://aclanthology.org/2023.trustnlp-1.28/).

## Confidence Estimates
We provide code to compute swapped pairs and expected calibration error (ECE) in `./src/`. Both scripts provide the option to compute swapped pairs and ECE using either the rand index (`rand`) or majority vote (`majority`) strategies, discussed in the paper. 

ECE and swapped pairs results are saved to `./data/statistics/`. 

Example command for running `compute_swapped_pairs.py` for the T0++ model using the majority vote strategy: 

    python compute_swapped_pairs.py \  
      --OIP_input_dir '../../data/predictions/OIP_prompt/T0/cos_e' \  
      --T0_prompt_input_dir '../../data/predictions/T0_prompts/T0/cos_e' \  
      --paraphrase_input_dir '../../data/predictions/paraphrase/T0/cos_e' \  
      --output_dir '../../data/statistics/swapped_pairs/majority' \  
      --model_name T0 \  
      --probability_col log_probabilities \  
      --sorting_algorithm mergesort \  
      --rank_by_rand_or_majority majority

Example command for running `compute_ece.py` for the T0++ model using the rand index strategy:

    python compute_ece.py \  
      --OIP_input_dir '../../data/predictions/OIP_prompt/T0/cos_e' \  
      --T0_prompt_input_dir '../../data/predictions/T0_prompts/T0/cos_e' \  
      --paraphrase_input_dir '../../data/predictions/paraphrase/T0/cos_e' \  
      --output_dir '../../data/statistics/ece' \  
      --model_name T0 \  
      --probability_col log_probabilities \  
      --sorting_algorithm mergesort \  
      --rank_by_rand_or_majority rand 

## Prediction Data

For demonstration purposes, we provide examples of the JSONL data files that `compute_ece.py` and `compute_swapped_pairs.py` expect in `./data/predictions/`. We include data for the three kinds of prompts: 
* `OIP_prompt`: the dataset's originally intended prompt, which we refer to as the Single, Human Prompt prompt in the paper,
* `T0_prompts`: the prompts written for the dataset when [Sanh et al. (2021)](https://arxiv.org/abs/2110.08207)  trained T0++, which we refer to as the Multiple, Human Prompts in the paper,
* `paraphrase`: the prompts that we automatically generate according to the method in the paper; we refer to these as the Automatically Generated Prompts in the paper. 

Note that the prediction data provided in this repository is a sample of the predictions used in the paper. We provide a small set of predictions from T0++, GPT davinci, and Flan-T5-XXL for one dataset ([cos_e/v1.11](https://huggingface.co/datasets/cos_e)), for a subset of inputs from the dataset, and a subset of prompts. 

## Output

ECE and swapped pairs results are saved to `./data/statistics/ece/`.

An example output CSV for ECE using T0++ and the rand index strategy is found at `./data/statistics/ece/rand/summary/T0.csv`:

    Dataset,OIP ECE,T0 Prompt ECE,Paraphrase ECE,T0 Prompt ECE < OIP ECE,Paraphrase ECE < OIP ECE,Paraphrase ECE < T0 Prompt ECE
    cos_e,2.4400360420346257,1.05856421739057,12.591883913675945,True,False,False

If `compute_ece.py` for T0++ and `rand` had been run on multiple datasets, then multiple datasets would be listed in the CSV.

## References
If you use the materials in this repository, please use the following citation:

    @inproceedings{portillowightman2023strength,
	    title={Strength in Numbers: Estimating Confidence of Large Language Models by Prompt Agreement},
	    author={Portillo Wightman, Gwenyth and DeLucia, Alexandra and Dredze, Mark},
	    booktitle={Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023)},
	    pages={326--362},
	    year={2023}
    }

