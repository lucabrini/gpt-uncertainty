import csv
import json
import numpy as np
import pandas as pd
import os
from sklearn.metrics import log_loss

data_path = "data/generation/8_mcrae/dialogues(gpt4o)_app1_onllama3_k5.csv"
filename = "data/generation/8_mcrae/entropy(gpt4o)_app1_onllama3_k5_apocalypse_cleaned.csv"

to_clean = True
to_apocalypse = True


def main():
    rf = open(data_path, 'r', newline='')
    reader = csv.DictReader(rf, delimiter=",")
    zeros_list = np.zeros(8)

    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    if file_exists:
        df = pd.read_csv(filename)
        if 'cross_entropy' not in df.columns:
            df['cross_entropy'] = np.nan
    else:
        df = pd.DataFrame(columns=["dialogue_id", "intra_dialogue_id", "cross_entropy"])

    cross_entropy_column = []

    previous_scores = []
    for row in reader:
        raw_scores = row["candidates_scores"].replace('\'', '"')
        raw_distribution = row["p_distribuition"].replace('\'', '"')
        target = row["target"]

        if len(raw_scores) != 0:
            json_scores = json.loads(raw_scores)
            json_distribution = json.loads(raw_distribution)

            scores = list(json_scores.values())
            distribution = list(json_distribution.values())

            if row["intra_dialogue_id"] == "0":
                previous_scores = scores

            if to_clean:
                distribution, scores = clean_scores(scores, previous_scores)
                previous_scores = scores

            # Calculate cross-entropy
            cross_entropy = calculate_cross_entropy(distribution, target, json_scores)

            '''if np.array_equal(zeros_list, distribution):
                print(zeros_list, distribution)
                if to_apocalypse:
                    # Set cross-entropy to an invalid value to be able to filter it later
                    cross_entropy = None
                else:
                    # Otherwise, set cross-entropy to max value
                    cross_entropy = None'''
            cross_entropy = round(cross_entropy, 4)
        else:
            cross_entropy = np.nan
        cross_entropy_column.append(cross_entropy)

        '''new_row = pd.DataFrame([{
            "dialogue_id": row["dialogue_id"],
            "intra_dialogue_id": row["intra_dialogue_id"],
            "cross_entropy": cross_entropy
        }])
        df = pd.concat([df, new_row], ignore_index=True)'''
    df["cross_entropy"] = cross_entropy_column
    rf.close()

    # Write the updated dataframe back to the file
    df.to_csv(filename, index=False)


def calculate_cross_entropy(distribution, target, scores_dict):
    """
    Calculate the cross-entropy given the predicted probabilities and the target item.
    :param distribution: A list of probabilities predicted by the model for the items.
    :param target: The target item which is the correct item.
    :param scores_dict: The dictionary containing the item scores.
    :return: The cross-entropy.
    """
    # Find the index of the target item
    target_index = list(scores_dict.keys()).index(target)
    true_distribution = np.zeros(len(distribution))
    true_distribution[target_index] = 1

    cross_entropy = log_loss([true_distribution], [distribution])
    print(np.argmax(true_distribution))
    print(distribution)
    print(cross_entropy)
    return cross_entropy


def clean_scores(current_scores, previous_scores, samples=5):
    """
    Clean scores by excluding the current candidates which have been excluded in the previous steps.
    :param current_scores: List of current scores.
    :param previous_scores: List of previous scores.
    :param samples: Number of samples to consider (default is 5).
    :return: Cleaned distribution and scores.
    """
    cleaned_scores = []

    for i, ith_score in enumerate(current_scores):
        if previous_scores[i] == 0:
            cleaned_scores.append(0)
        else:
            cleaned_scores.append(ith_score)

    scores_sum = sum(cleaned_scores)
    cleaned_distribution = []

    for ith_score in cleaned_scores:
        try:
            normalized_ith_score = round(ith_score / scores_sum, 4)
        except ZeroDivisionError:
            normalized_ith_score = 0
        cleaned_distribution.append(normalized_ith_score)

    return cleaned_distribution, cleaned_scores


main()