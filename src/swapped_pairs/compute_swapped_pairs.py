from glob import glob
import os
from os import listdir
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.metrics.cluster import rand_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute stats for swapped pairs.")
    parser.add_argument(
        "--OIP_input_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--T0_prompt_input_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--paraphrase_input_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./',
        help="Path where results will be saved.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='t0',
        help="Which model predictions are from (e.g., 'T0', 'GPT3', 'flan').",
        required=True,
    )
    parser.add_argument(
        "--probability_col",
        type=str,
        default='log_probabilities',
        help="Name of column with probabilities ('probabilities', 'log_probabilities').",
        required=True,
    )
    parser.add_argument(
        "--sorting_algorithm",
        type=str,
        default='quicksort',
        help="Pandas sort_values() sorting algorithm (quicksort, mergesort, etc.)",
        required=True,
    )
    parser.add_argument(
        "--rank_by_rand_or_majority",
        type=str,
        default='majority',
        help="Sort by rand index 'rand' or by majority label 'majority'.",
        required=True
    )

    args = parser.parse_args()
    return args


def convert_agreement_to_ri_lists(agreement_list):
    # take sum over values in agreement_list
    # make a list of 1s with that length
    num_prompts = sum(agreement_list)
    actual_label_list = [1] * num_prompts

    # keep some kind of curr_count thing to indicate what value we are filling the list with
    # iterate over AL
    # if AL[i] is not zero
        # append AL[i] number of curr_count to the list
    predicted_label_list = []
    curr_label_num = 1
    for i in range(len(agreement_list)):
        if agreement_list[i] > 0:
            new_labels = agreement_list[i] * [curr_label_num]
            predicted_label_list += new_labels
            curr_label_num += 1

    # return as a tuple
    ri_lists = (predicted_label_list, actual_label_list)
    return ri_lists


"""
agreement_list: str type, e.g., ("[1, 1, 1, 1]"
"""
def compute_rand_index(agreement_list_as_str):
    # Remove [, ], and space characters. Convert values to integers.
    pred_list = agreement_list_as_str.replace('[', '').replace(']', '').replace(' ', '').split(',')
    for idx in range(len(pred_list)):
        pred_list[idx] = int(pred_list[idx])
    rand_index_lists = convert_agreement_to_ri_lists(pred_list)
    rand_index = rand_score(rand_index_lists[0], rand_index_lists[1])
    return rand_index


def compute_swapped_pairs(arr):
    inv_count = 0
    n = len(arr)

    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] < arr[j]:
                inv_count += 1

    return inv_count


def prepare_agreement_table(args, input_dir):
    # Makes a table with two new columns:
        # Rand index
        # Accuracy of majority prediction (accounts for tied-majority predictions) (majority label(s) correctness)

    PROB_COLUMN_NAME = args.probability_col

    if 'gpt' in args.model_name:
        ID_COLUMN_NAME = 'id'
        TARGET_INDEX_COLUMN_NAME = 'target_index'
        CHOICES_COLUMN_NAME = 'untokenized_answer_choices'
        # Use answer choice index as prediction value
        # This avoids texts and IDs that refer to the same choice from being counted as different answers
        PREDICTION_COLUMN_NAME = 'prediction_as_answer_choice_index'
    else:
        ID_COLUMN_NAME = 'context_id'
        TARGET_COLUMN_NAME = 'target'
        PREDICTION_COLUMN_NAME = 'prediction'

    # Get files from input directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
                  os.path.isfile(os.path.join(input_dir, f))]

    num_prompts = len(files)
    dataset_name = input_dir.split('/')[-1]

    if input_dir == args.T0_prompt_input_dir:
        output_file_path = os.path.join(args.output_dir, args.rank_by_rand_or_majority, 'T0_prompts', args.model_name)
        # os.makedirs(output_file_path, exist_ok=True)
        final_df_output_file_name = f'{dataset_name}_T0_prompt_swapped_pairs.csv'
        final_df_output_file_path = os.path.join(output_file_path, final_df_output_file_name)
        # For gpt, add a file to save output of unfiltered dataframe
        if 'gpt' in args.model_name:
            unfiltered_df_output_file_path = os.path.join(args.output_dir, 'T0_prompts', args.model_name, 'unfiltered_data')
            # os.makedirs(unfiltered_df_output_file_path, exist_ok=True)
            unfiltered_data_output_file_name = f'{dataset_name}_T0_prompt_swapped_pairs_unfiltered.csv'
            unfiltered_df_output_file_path = os.path.join(unfiltered_df_output_file_path, unfiltered_data_output_file_name)
    else:  # input_dir == args.paraphrase_input_dir
        if dataset_name == '':
            dataset_name = input_dir.split('/')[-2]
        output_file_path = os.path.join(args.output_dir, args.rank_by_rand_or_majority, 'paraphrase', args.model_name)
        # os.makedirs(output_file_path, exist_ok=True)
        output_file_name = f'{dataset_name}_paraphrase_swapped_pairs.csv'
        final_df_output_file_path = os.path.join(output_file_path, output_file_name)

    print(f"Processing files in directory {input_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Num files/prompts: {num_prompts}")
    print(f"File names: {files}")

    # `dfs` is a dictionary that holds (key, value) pairs where key is the name of a file and
    # value is the dataframe for that file.
    dfs = {}

    # Get list of ids and targets from the first file
    with open(files[0]) as f:
        data = [json.loads(line) for line in f]
    single_file_df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')
    print(single_file_df.columns)
    original_dataset_size = len(single_file_df)

    num_examples = len(single_file_df)

    ids_list = single_file_df[ID_COLUMN_NAME].tolist()
    # Get a list of targets and the number of answer choices
    if 'gpt' in args.model_name:
        targets_list = []
        target_index_list = []
        for index, row in single_file_df.iterrows():
            target_index = row[TARGET_INDEX_COLUMN_NAME]
            choices = row[CHOICES_COLUMN_NAME]
            new_target = choices[target_index]
            targets_list.append(new_target)
            target_index_list.append(target_index)
            # Get number of answer choices by taking the length of choices list (for gpt models)
            num_answer_choices = len(choices)
    else:
        targets_list = single_file_df[TARGET_COLUMN_NAME].tolist()
        # Get number of answer choices by taking the length of a probability list (for non-gpt models)
        probability_list_example = single_file_df[args.probability_col].tolist()[0]
        target_index_list = single_file_df[TARGET_COLUMN_NAME].tolist()
        num_answer_choices = len(probability_list_example)
    print(f"Num answer choices: {num_answer_choices}")

    # Make dataframe with IDs and targets
    full_df = pd.DataFrame({'id': ids_list, 'target': targets_list, 'target_index': target_index_list})

    prompt_names = []

    for (i, file) in enumerate(files):
        print(f"Processing {file}")

        # Isolate the prompt name from the file name
        prompt_name = file.split('/')[-1]
        prompt_name = prompt_name.replace('.jsonl', '')
        prompt_name = prompt_name.replace(f'{args.model_name}_', '')
        prompt_names.append(prompt_name)
        print(f"Working with file {file}, with prompt name {prompt_name}")

        # Read file into dataframe
        with open(file) as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')

        # Save data from iterating over rows to dictionaries
        # Key is id (like in ids_list) and values are text, tokens, logprobs
        prediction_dict = {}
        probs_dict = {}

        for (index, row) in df.iterrows():
            id = row[ID_COLUMN_NAME]
            prediction = row[PREDICTION_COLUMN_NAME]
            probs = row[PROB_COLUMN_NAME]

            prediction_dict[id] = prediction
            probs_dict[id] = probs

        # Need to add data to these lists in the same order that they are in for ids_list and targets_list
        prediction_list = []
        correctness_list = []
        probs_list = []

        for (i, id) in enumerate(ids_list):
            prediction = prediction_dict[id]
            prediction_list.append(prediction)

            probs = probs_dict[id]
            probs_list.append(probs)

            if prediction == targets_list[i]:
                correctness_list.append(1)
            else:
                correctness_list.append(0)

        full_df['prediction_' + prompt_name] = prediction_list
        full_df['correctness_' + prompt_name] = correctness_list
        full_df['probs_' + prompt_name] = probs_list

    '''
    Use .startswith to get lists of column names contain predictions, probabilities, and correctness.
    Use .startswith because there are multiple columns that begin with those prefixes, since we 
    renamed columns above to avoid problems when merging DFs.
    '''
    prediction_cols = [col for col in full_df if col.startswith('prediction')]
    correctness_cols = [col for col in full_df if col.startswith('correctness')]
    prob_cols = [col for col in full_df if col.startswith('probs')]

    # Make empty lists that we will fill and then use as new columns of `df_all`
    agreement_level_list = []  # used to group rows. each item is list of agreement (e.g., [0, 0, 1, 4]) as a string
    correct_label_was_predicted_list = []  # for coverage metric
    accuracy_across_prompts_list = []  # for accuracy metric
    majority_correctness_proportion_list = []  # for majority metric
    prediction_probability_dict_list = []
    max_average_probability_of_majority_labels_list = []
    accuracy_of_maj_label_with_max_prob_list = []
    majority_label_with_max_prob_list = []

    '''
    For each instance in the dataset, create lists that contain the prediction and correctness
    value (whether or not that label matches the target)
    '''
    for (index, row) in full_df.iterrows():
        prediction_values = {}
        correctness_values = {}
        prediction_probability_dict = {}

        # Iterate over columns that contain predictions, probs, logprobs, and correctness
        # First iterate over predictions
        for col in prediction_cols:
            prediction = row[col]
            # If the prediction is not already in the prediction_values dictionary, add it with a count of 1
            # Otherwise, increment the count
            if not np.isnan(prediction):  # Predictions are NaN when they could not be parsed to match an answer choice
                if row[col] not in prediction_values.keys():
                    prediction_values[prediction] = 1
                else:
                    prediction_values[prediction] += 1

            curr_probability_col = col.replace('prediction_', 'probs_')
            # Get probability
            if 'gpt' in args.model_name:
                probability = row[curr_probability_col]
            else:
                probability_list = row[curr_probability_col]
                probability = probability_list[prediction]
            # Save probability to dictionary
            if prediction not in prediction_probability_dict:
                prediction_probability_dict[prediction] = [probability]  # Initialize the probability list
            else:
                prediction_probability_dict[prediction].append(probability)
        prediction_probability_dict_list.append(prediction_probability_dict)

        # Now iterate over the correctness of the text for each prompt
        for col in correctness_cols:
            correctness = row[col]
            # Obtain the text corresponding to this correctness column
            prediction_col = col.replace('correctness', 'prediction')
            # Save this (prediction, correctness) pair to the correctness dictionary
            correctness_values[row[prediction_col]] = correctness

        # Find agreement level column for each row
        # Make a sorted agreement label for each row
        # Get the values from the prediction_values
        agreement_values = list(prediction_values.values())
        agreement_values.sort(reverse=True)
        # Pack the agreement values list with 0s until it has length=num_prompts
        # (similar to https://stackoverflow.com/questions/70406616/fill-lists-in-list-with-zeros-if-their-length-less-than)
        zero_list = [0] * num_answer_choices
        agreement_values = agreement_values[:num_prompts] + zero_list[len(agreement_values):]

        # *** If agreement_values (a list) cannot be saved in the dataframe, use agreement level string
        # Convert to string -- https://stackoverflow.com/questions/17796446/convert-a-list-to-a-string-and-back
        agreement_level_string = json.dumps(agreement_values)
        agreement_level_list.append(agreement_level_string)

        # Find majority correctness of this row
        unsorted_agreement_values = list(prediction_values.values())
        if len(unsorted_agreement_values) == 0:
            correct_label_was_predicted_list.append(None)
            accuracy_across_prompts_list.append(None)
            majority_correctness_proportion_list.append(None)
            max_average_probability_of_majority_labels_list.append(None)
            accuracy_of_maj_label_with_max_prob_list.append(None)
            majority_label_with_max_prob_list.append(None)
            continue

        max_level_of_agreement = max(unsorted_agreement_values)
        # See if multiple labels have the max agreement level (i.e., there are multiple majority labels)
        num_majority_labels = 0
        majority_labels_list = []
        for key in prediction_values.keys():
            if prediction_values[key] == max_level_of_agreement:
                num_majority_labels += 1
                majority_labels_list.append(key)

        if 'gpt' in args.model_name:
            target = int(row['target_index'])
        else:
            target = row['target']

        # Compare majority label(s) and target
        if num_majority_labels == 1:
            if majority_labels_list[0] == target:
                majority_correctness_proportion = 1.0
            else:
                majority_correctness_proportion = 0
        else:  # More than one majority label
            # See if any of the majority labels match the target
            majority_label_matches_target_bool = False
            for label in majority_labels_list:
                label = int(label)
                if label == target:
                    majority_label_matches_target_bool = True
                    break

            # If majority_prediction_is_correct, that means that among the majority labels, one of those labels was correct.
            # So The expectation of randomly selecting a label from the list of majority labels and that selected label being true
            # is equivalent to 1/(num_majority_prompts)
            if majority_label_matches_target_bool:
                majority_correctness_proportion = 1 / float(num_majority_labels)
            # Else, the correct label is not among the majority labels, so if we randomly pick a label from a list of all wrong
            # labels, the chances of getting a correct label is 0.
            else:
                majority_correctness_proportion = 0

        majority_correctness_proportion_list.append(majority_correctness_proportion)

        # Find coverage of this row
        if 1 in correctness_values.values():
            coverage = 1
        else:
            coverage = 0
        correct_label_was_predicted_list.append(coverage)

        # Use `majority_labels_list` to calculate the average probability of the majority label(s)
        # Only save the maximum average probability (this is important if there are multiple majority labels)
        max_average_probability = -1000000
        for label in majority_labels_list:
            probability_list = prediction_probability_dict[label]
            average_probability = sum(probability_list) / len(probability_list)
            if average_probability > max_average_probability:
                max_average_probability = average_probability
                majority_label_with_max_prob = label
                if label == target:
                    accuracy_of_maj_label_with_max_prob = 1.0
                else:
                    accuracy_of_maj_label_with_max_prob = 0
        max_average_probability_of_majority_labels_list.append(max_average_probability)
        accuracy_of_maj_label_with_max_prob_list.append(accuracy_of_maj_label_with_max_prob)
        majority_label_with_max_prob_list.append(majority_label_with_max_prob)

        # `accuracy_across_prompts` is calculated for each text example
        # It is defined as the (total number of correct predictions) / (total number of prompts)
        # accuracy_across_prompts = prediction_values.count(target) / float(num_prompts)

        if target in prediction_values.keys():
            count_target_was_predicted = prediction_values[target]
            accuracy = count_target_was_predicted / float(num_prompts)
        else:
            accuracy = 0.0
        accuracy_across_prompts_list.append(accuracy)

        # report results: group by agreement level (which is sorted prediction_values.values) and output accuracy, majority, coverage

    # After iterating over dataframe, add new columns with agreement level, whether correct label was predicted, accuracy, and majority
    full_df['agreement_level_as_str'] = agreement_level_list
    full_df['correct_label_was_predicted'] = correct_label_was_predicted_list
    full_df['accuracy_across_prompts'] = accuracy_across_prompts_list
    full_df['majority_correctness_proportion'] = majority_correctness_proportion_list
    full_df['prediction_probability_dict'] = prediction_probability_dict_list
    full_df['majority_label_with_max_prob'] = majority_label_with_max_prob_list
    full_df['max_average_probability_of_majority_labels'] = max_average_probability_of_majority_labels_list
    full_df['accuracy_of_maj_label_with_max_prob'] = accuracy_of_maj_label_with_max_prob_list

    # Add column with rand score for each row
    rand_index_list = []
    for index, row in full_df.iterrows():
        agreement_list = row['agreement_level_as_str']
        rand_index = compute_rand_index(agreement_list)
        rand_index_list.append(rand_index)
    full_df['rand_index'] = rand_index_list

    if args.rank_by_rand_or_majority == 'majority':
        full_df = full_df.sort_values(by='max_average_probability_of_majority_labels', ascending=False, kind=args.sorting_algorithm)
    else:  # Sort by rand score
        full_df = full_df.sort_values(by='rand_index', ascending=False, kind=args.sorting_algorithm)

    # For gpt3, drop rows that are missing predictions
    if 'gpt' in args.model_name:
        # First save unfiltered dataframe
        # full_df.to_csv(unfiltered_df_output_file_path, index=False)

        num_valid_predictions_list = []
        for index, row in full_df.iterrows():
            agreement_list_as_str = row['agreement_level_as_str']
            # Convert agreement_level_as_str to a list of integer
            agreement_list = agreement_list_as_str.replace('[', '').replace(']', '').replace(' ', '').split(',')
            for idx in range(len(agreement_list)):
                agreement_list[idx] = int(agreement_list[idx])

            num_valid_predictions = sum(agreement_list)
            num_valid_predictions_list.append(num_valid_predictions)

        full_df['num_valid_predictions'] = num_valid_predictions_list

        # full_df = full_df[full_df['num_valid_predictions'] == num_prompts]]
        full_df = full_df[full_df['num_valid_predictions'] == num_prompts]
        full_df = full_df.reset_index()

    print(full_df.head())

    # full_df.to_csv(final_df_output_file_path, index=False)
    return full_df, original_dataset_size


def prepare_OIP_table(args, input_dir):
    if 'gpt' in args.model_name:
        TARGET_COLUMN_NAME = 'answer'
        PREDICTION_COLUMN_NAME = 'parsed_prediction'
        PROBABILITY_COLUMN_NAME = args.probability_col
        PRED_IS_CORRECT_COLUMN_NAME = 'parsed_prediction_is_correct'
    else:
        TARGET_COLUMN_NAME = 'target'
        PREDICTION_COLUMN_NAME = 'prediction'
        PROBABILITY_COLUMN_NAME = args.probability_col

    # Get OIP file from input directory
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if
             os.path.isfile(os.path.join(input_dir, f))]
    file = files[0]

    print(file)

    dataset_name = input_dir.split('/')[-1]

    output_file_path = os.path.join(args.output_dir, 'OIP', args.model_name)
    # os.makedirs(output_file_path, exist_ok=True)
    output_file_name = f'{dataset_name}_OIP_swapped_pairs.csv'
    output_file_path = os.path.join(output_file_path, output_file_name)

    # Get list of ids and targets from the first file
    with open(file) as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')
    print(df.columns)

    pred_probability_list = []
    correctness_list = []
    for index, row in df.iterrows():
        prediction = row[PREDICTION_COLUMN_NAME]
        target = row[TARGET_COLUMN_NAME]

        if 'gpt' in args.model_name:
            pred_probability = row[PROBABILITY_COLUMN_NAME]
        else:
            probabilities = row[PROBABILITY_COLUMN_NAME]
            pred_probability = probabilities[prediction]

        pred_probability_list.append(pred_probability)

        if 'gpt' in args.model_name:
            correctness_val = row[PRED_IS_CORRECT_COLUMN_NAME]
            correctness_list.append(correctness_val)
        else:
            if target == prediction:
                correctness_list.append(1)
            else:
                correctness_list.append(0)

    df['pred_probability'] = pred_probability_list
    df['correctness'] = correctness_list

    df = df.sort_values(by='pred_probability', ascending=False, kind=args.sorting_algorithm)

    # For gpt3, drop rows that are missing predictions
    if 'gpt' in args.model_name:
        num_valid_predictions_list = []
        for index, row in df.iterrows():
            if np.isnan(row['pred_probability']):
                print('FOUND NAN')
                num_valid_predictions_list.append(0)
            else:
                num_valid_predictions_list.append(1)

        df['num_valid_predictions'] = num_valid_predictions_list

        # Keep rows with one valid prediction (drop rows with zero valid predictions)
        df = df[df['num_valid_predictions'] == 1]
        df = df.reset_index()

    if 'gpt' in args.model_name:
        df = df.drop(['prompt', 'api_call_parameters.model', 'api_call_parameters.prompt', 'api_call_parameters.stop', 'api_call_parameters.temperature', 'api_call_parameters.logprobs', 'gpt3_response.id', 'gpt3_response.object', 'gpt3_response.created', 'gpt3_response.model', 'gpt3_response.choices', 'gpt3_response.usage.prompt_tokens', 'gpt3_response.usage.completion_tokens', 'gpt3_response.usage.total_tokens'], axis=1)

    # df.to_csv(output_file_path, index=False)
    return df


def main():
    args = parse_args()

    if args.rank_by_rand_or_majority != 'majority' and args.rank_by_rand_or_majority != 'rand':
        print("Unrecognized rank_by_rand_or_majority argument.")
        exit()

    if args.model_name == 'gpt':
        # Prepare dataframes
        OIP_df = prepare_OIP_table(args, args.OIP_input_dir)
        T0_prompt_df, original_dataset_len = prepare_agreement_table(args, args.T0_prompt_input_dir)

        # Only keep rows of the OIP dataframe that contain an ID that is present in the T0_prompt_df
        T0_ids_to_keep = T0_prompt_df['id'].tolist()
        OIP_ids_to_keep = OIP_df['id'].tolist()
        ids_to_keep = list(set(T0_ids_to_keep) & set(OIP_ids_to_keep))
        # Keep relevant ids in OIP df; drop irrelevant rows
        for index, row in OIP_df.iterrows():
            curr_id = row['id']
            if curr_id not in ids_to_keep:
                OIP_df.drop(index, inplace=True)
        # Keep relevant ids in T0 df; drop irrelevant rows
        for index, row in T0_prompt_df.iterrows():
            curr_id = row['id']
            if curr_id not in ids_to_keep:
                T0_prompt_df.drop(index, inplace=True)

        # Get length of dataset with only rows that have the same number of valid parsed predictions (a prediction choice
            # that matches an answer choice) as the number of prompts for the dataset
        filtered_dataset_size = len(OIP_df)

        # Compute swapped_pairs
        OIP_swapped_pairs = compute_swapped_pairs(OIP_df['correctness'].tolist())
        T0_prompt_swapped_pairs = compute_swapped_pairs(T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist())

        # Compute swapped pairs with normalization for macroaverage
        # First, divide by size of filtered dataset
        OIP_swapped_pairs_ds = compute_swapped_pairs(OIP_df['correctness'].tolist()) * (1.0 / filtered_dataset_size)
        T0_prompt_swapped_pairs_ds = compute_swapped_pairs(T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / filtered_dataset_size)
        # Now divde by square of the size of filtered dataset
        OIP_swapped_pairs_s2 = compute_swapped_pairs(OIP_df['correctness'].tolist()) * (1.0 / (filtered_dataset_size * filtered_dataset_size))
        T0_prompt_swapped_pairs_s2 = compute_swapped_pairs(
            T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / (filtered_dataset_size * filtered_dataset_size))

        T0_prompt_less_than_OIP = T0_prompt_swapped_pairs < OIP_swapped_pairs

        summary_output_dir = os.path.join(args.output_dir, 'summary')
        os.makedirs(summary_output_dir, exist_ok=True)
        output_file_path = os.path.join(summary_output_dir, f'{args.model_name}.csv')

        with open(output_file_path, "a") as myfile:
            if os.stat(output_file_path).st_size == 0:
                myfile.write(
                    f'Dataset,OIP Swapped Pairs,T0 Prompt Swapped Pairs,OIP Swapped Pairs (1/dataset_size),T0 Prompt Swapped Pairs (1/dataset_size),OIP Swapped Pairs (1/(dataset_size^2)),T0 Prompt Swapped Pairs (1/(dataset_size^2))\n')
            myfile.write(
                f'{args.OIP_input_dir.split("/")[-1]},{OIP_swapped_pairs},{T0_prompt_swapped_pairs},{OIP_swapped_pairs_ds},{T0_prompt_swapped_pairs_ds},{OIP_swapped_pairs_s2},{T0_prompt_swapped_pairs_s2}\n')

        # Save stats about dataset length
        # gpt_stats_output_dir = os.path.join(args.output_dir, 'gpt3_stats')
        # os.makedirs(gpt_stats_output_dir, exist_ok=True)
        # output_file_path = os.path.join(gpt_stats_output_dir, f'dataset_sizes.csv')

        # with open(output_file_path, "a") as myfile:
        #     if os.stat(output_file_path).st_size == 0:
        #         myfile.write(
        #             f'Dataset,Original Size,Size Filtered for Valid Predictions,Proportion\n')
        #     myfile.write(
        #         f'{args.OIP_input_dir.split("/")[-1]},{original_dataset_len},{filtered_dataset_size},{filtered_dataset_size / float(original_dataset_len)}\n')

    else:
        # Prepare dataframes
        OIP_df = prepare_OIP_table(args, args.OIP_input_dir)
        T0_prompt_df, original_dataset_len = prepare_agreement_table(args, args.T0_prompt_input_dir)
        paraphrase_df, original_dataset_len = prepare_agreement_table(args, args.paraphrase_input_dir)

        filtered_dataset_size = len(T0_prompt_df)

        # Compute swapped_pairs
        OIP_swapped_pairs = compute_swapped_pairs(OIP_df['correctness'].tolist())
        T0_prompt_swapped_pairs = compute_swapped_pairs(T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist())
        paraphrase_swapped_pairs = compute_swapped_pairs(paraphrase_df['accuracy_of_maj_label_with_max_prob'].tolist())

        # Compute swapped pairs with normalization for macroaverage
        # First, divide by size of filtered dataset
        OIP_swapped_pairs_ds = compute_swapped_pairs(OIP_df['correctness'].tolist()) * (1.0 / filtered_dataset_size)
        T0_prompt_swapped_pairs_ds = compute_swapped_pairs(T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / filtered_dataset_size)
        paraphrase_swapped_pairs_ds = compute_swapped_pairs(paraphrase_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / filtered_dataset_size)
        # Now divde by square of the size of filtered dataset
        OIP_swapped_pairs_s2 = compute_swapped_pairs(OIP_df['correctness'].tolist()) * (1.0 / (filtered_dataset_size * filtered_dataset_size))
        T0_prompt_swapped_pairs_s2 = compute_swapped_pairs(
            T0_prompt_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / (filtered_dataset_size * filtered_dataset_size))
        paraphrase_swapped_pairs_s2 = compute_swapped_pairs(
            paraphrase_df['accuracy_of_maj_label_with_max_prob'].tolist()) * (1.0 / (filtered_dataset_size * filtered_dataset_size))

        # T0_prompt_less_than_OIP = T0_prompt_swapped_pairs < OIP_swapped_pairs
        # paraphrase_less_than_OIP = paraphrase_swapped_pairs < OIP_swapped_pairs
        # paraphrase_less_than_T0_prompt = paraphrase_swapped_pairs < T0_prompt_swapped_pairs

        summary_output_dir = os.path.join(args.output_dir, 'summary')
        os.makedirs(summary_output_dir, exist_ok=True)
        output_file_path = os.path.join(summary_output_dir, f'{args.model_name}.csv')

        with open(output_file_path, "a") as myfile:
            if os.stat(output_file_path).st_size == 0:
                myfile.write(f'Dataset,OIP Swapped Pairs,T0 Prompt Swapped Pairs,Paraphrase Swapped Pairs,OIP Swapped Pairs (1/dataset_size),T0 Prompt Swapped Pairs (1/dataset_size),Paraphrase Swapped Pairs (1/dataset_size),OIP Swapped Pairs (1/(dataset_size^2)),T0 Prompt Swapped Pairs (1/(dataset_size^2)),Paraphrase Swapped Pairs (1/(dataset_size^2))\n')
            myfile.write(f'{args.OIP_input_dir.split("/")[-1]},{OIP_swapped_pairs},{T0_prompt_swapped_pairs},{paraphrase_swapped_pairs},{OIP_swapped_pairs_ds},{T0_prompt_swapped_pairs_ds},{paraphrase_swapped_pairs_ds},{OIP_swapped_pairs_s2},{T0_prompt_swapped_pairs_s2},{paraphrase_swapped_pairs_s2}\n')

        # # Save stats about dataset length
        # stats_output_dir = os.path.join(args.output_dir, f'other_model_stats')
        # os.makedirs(stats_output_dir, exist_ok=True)
        # output_file_path = os.path.join(stats_output_dir, f'{args.model_name}_stats.csv')
        #
        # with open(output_file_path, "a") as myfile:
        #     if os.stat(output_file_path).st_size == 0:
        #         myfile.write(
        #             f'Dataset,Original Size,Size Filtered for Valid Predictions,Proportion\n')
        #     myfile.write(
        #         f'{args.OIP_input_dir.split("/")[-1]},{original_dataset_len},{filtered_dataset_size},{filtered_dataset_size / float(original_dataset_len)}\n')


if __name__ == "__main__":
    main()
