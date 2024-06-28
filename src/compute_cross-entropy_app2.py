import csv
import json
import numpy as np
import ast
from collections import Counter
from sklearn.metrics import log_loss
import os
import pandas as pd

data_path = "data/generation/8_mcrae/dialogues(gpt3)_app2_onllama3_k5.csv"
filename = "entropy(gpt3)_app2_onllama3_k5_apocalypse_cleaned"

to_clean = True


def main():
    rf = open(data_path, 'r', newline='')
    reader = csv.DictReader(rf, delimiter=",")

    # Assuming we know the possible candidates for all dialogues
    all_possible_candidates = set()

    for row in reader:
        raw_distribution = row["p_distribuition"].replace('\'', '"')
        json_distribution = json.loads(raw_distribution)
        all_possible_candidates.update(json_distribution.keys())

    all_possible_candidates = sorted(list(all_possible_candidates))

    rf.seek(0)  # Reset reader to the beginning of the file

    # Check if the file already exists
    file_exists = os.path.isfile(f"data/generation/8_mcrae/{filename}.csv")

    if file_exists:
        df = pd.read_csv(f"data/generation/8_mcrae/{filename}.csv")
        if 'cross_entropy' not in df.columns:
            df['cross_entropy'] = np.nan
    else:
        df = pd.DataFrame(columns=["dialogue_id", "intra_dialogue_id", "entropy", "cross_entropy"])

    cross_entropy_column = []

    for row in reader:
        raw_scores = row["candidates_scores"].replace('\'', '"')
        raw_distribution = row["p_distribuition"].replace('\'', '"')
        row_k_candidates = row["candidates"].replace('\'', '"').replace('set()', '{None}')
        target = row["target"]

        if raw_scores and raw_distribution:
            try:
                json_scores = json.loads(raw_scores)
                json_distribution = json.loads(raw_distribution)
            except json.JSONDecodeError:
                continue

            elements = row_k_candidates[1:-1].split('}, {')
            list_of_sets = [
                set(ast.literal_eval('{' + element.strip('{}') + '}')) if element.strip('{}') != 'None' else None for
                element in elements]
            counter = Counter()
            for s in list_of_sets:
                counter.update(s)
            k_candidates = dict(counter)

            scores = list(json_scores.values())
            distr = list(json_distribution.values())
            list_candidates = list(json_distribution.keys())

            if row["intra_dialogue_id"] == "0":
                previous_scores = scores
                previous_distr = distr
                previous_list_candidates = list_candidates

            if to_clean:
                distr, list_candidates = clean_scores(list_candidates, previous_list_candidates, k_candidates)
                previous_list_candidates = list_candidates

            full_distribution = {candidate: 0.0 for candidate in all_possible_candidates}
            full_distribution.update(json_distribution)
            distr = list(full_distribution.values())

            cross_entropy = calculate_cross_entropy(distr, target, full_distribution)
            if all(v == 0 for v in distr):
                cross_entropy = np.nan
            cross_entropy = round(cross_entropy, 4)
            print(cross_entropy)
        else:
            cross_entropy = np.nan

        cross_entropy_column.append(cross_entropy)

    df['cross_entropy'] = cross_entropy_column
    rf.close()

    # Write the updated dataframe back to the file
    df.to_csv(f"data/generation/8_mcrae/{filename}.csv", index=False)


def calculate_cross_entropy(distribution, target, scores_dict):
    """
    Calculate the cross-entropy given the predicted probabilities and the target item.
    :param distribution: A list of probabilities predicted by the model for the items.
    :param target: The target item which is the correct item.
    :param scores_dict: The dictionary containing the item scores.
    :return: The cross-entropy.
    """
    target_index = list(scores_dict.keys()).index(target)
    true_distribution = np.zeros(len(distribution))
    true_distribution[target_index] = 1

    cross_entropy = log_loss([true_distribution], [distribution])
    return cross_entropy


def clean_scores(list_candidates, previous_list_candidates, result_dict):
    """
    Clean scores by excluding the current candidates which have been excluded in the previous steps.
    :param list_candidates: List of current candidates.
    :param previous_list_candidates: List of previous candidates.
    :param result_dict: Dictionary containing the result scores.
    :return: Cleaned distribution and list of candidates.
    """
    new_candidates = [item for item in list_candidates if item not in previous_list_candidates]

    cleaned_distribution = {item: result_dict[item] for item in result_dict if item not in new_candidates}

    total_count = sum(cleaned_distribution.values())

    normalized_distribution = {item: round(count / total_count, 4) for item, count in cleaned_distribution.items()}

    new_list_candidates = [item for item in list_candidates if item not in new_candidates]

    return normalized_distribution.values(), new_list_candidates


main()