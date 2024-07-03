import csv
import json
import numpy as np
import ast
from collections import Counter
from sklearn.metrics import log_loss
import os
import pandas as pd

data_path = "data/generation/8_mcrae/dialogues(gpt3)_k_five_sim_app_gpt3.csv"
filename = "sbs_entropy(gpt3)_k_five_gpt3_sim_app_apocalypse_cleaned"
contrast_sets_path = "data/game_sets/8_mcrae/contrast_sets.json"

plot = True

to_clean = True


def main():
    rf = open(data_path, 'r', newline='')
    reader = csv.DictReader(rf, delimiter=",")

    # Load contrast sets from JSON file
    with open(contrast_sets_path, 'r') as f:
        contrast_sets = json.load(f)

    # Check if the file already exists
    file_exists = os.path.isfile(f"data/generation/8_mcrae/{filename}.csv")
    if file_exists:
        df = pd.read_csv(f"data/generation/8_mcrae/{filename}.csv")
        if 'cross_entropy' in df.columns:
            df = df.drop(columns=['cross_entropy'])
    else:
        df = pd.DataFrame(columns=["dialogue_id", "intra_dialogue_id", "entropy", "cross_entropy"])

    cross_entropy_column = []

    for row in reader:
        raw_scores = row["candidates_scores"].replace('\'', '"')
        raw_distribution = row["p_distribuition"].replace('\'', '"')
        row_k_candidates = row["candidates"].replace('\'', '"').replace('set()', '{None}')
        target = row["target"]
        dialogue_id = row["dialogue_id"]

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

            # Use the candidates specific to the current dialogue_id from contrast_sets.json
            all_possible_candidates = contrast_sets[dialogue_id]["items"]
            full_distribution = {candidate: 0.0 for candidate in all_possible_candidates}
            for candidate, value in json_distribution.items():
                if candidate in full_distribution:
                    full_distribution[candidate] = value
                else:
                    print(
                        f"Warning: Candidate {candidate} not found in all_possible_candidates for dialogue_id {dialogue_id}")

            distr = list(full_distribution.values())

            cross_entropy = calculate_cross_entropy(distr, target, full_distribution)
            if all(v == 0 for v in distr):
                cross_entropy = np.nan
            cross_entropy = round(cross_entropy, 4)
            print(cross_entropy)
        else:
            cross_entropy = np.nan

        cross_entropy_column.append({
            "dialogue_id": int(row["dialogue_id"]),
            "intra_dialogue_id": int(row["intra_dialogue_id"]),
            "cross_entropy": cross_entropy
        })

    rf.close()

    # Convert cross_entropy_column to a DataFrame
    cross_entropy_df = pd.DataFrame(cross_entropy_column)

    # Merge the new cross_entropy_df with the existing df
    if len(df)>0:
        df = pd.merge(df, cross_entropy_df, on=["dialogue_id", "intra_dialogue_id"], how="left")
    else:
        df = cross_entropy_df

    # Write the updated dataframe back to the file
    df.to_csv(f"data/generation/8_mcrae/{filename}.csv", index=False)

    # Check if the necessary columns exist in the DataFrame
    if 'entropy' in df.columns and 'cross_entropy' in df.columns:
        # Compute the correlation between 'entropy' and 'cross_entropy'
        correlation = df['entropy'].corr(df['cross_entropy'])
        print(f"The correlation between entropy and cross_entropy is: {correlation}")
    else:
        print("The DataFrame does not contain the required columns 'entropy' and 'cross_entropy'.")

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np

        # Ensure a compatible backend is used
        matplotlib.use('Agg')

        # Group by 'intra_dialogue_id' and calculate the mean and std for cross-entropy and entropy
        mean_cross_entropy_per_step = df.groupby('intra_dialogue_id')['cross_entropy'].mean()
        std_cross_entropy_per_step = df.groupby('intra_dialogue_id')['cross_entropy'].std()
        mean_entropy_per_step = df.groupby('intra_dialogue_id')['entropy'].mean()
        std_entropy_per_step = df.groupby('intra_dialogue_id')['entropy'].std()

        # Create a range for x-axis (dialogue steps)
        steps = range(len(mean_cross_entropy_per_step))

        # Plot the mean cross-entropy and entropy with standard deviation as confidence intervals
        plt.figure(figsize=(10, 6))

        # Plot mean cross-entropy
        plt.plot(steps, mean_cross_entropy_per_step, marker='o', linestyle='-', color='b', label='Cross-Entropy')
        plt.fill_between(steps,
                         mean_cross_entropy_per_step - std_cross_entropy_per_step,
                         mean_cross_entropy_per_step + std_cross_entropy_per_step,
                         color='b', alpha=0.2)

        # Plot mean entropy
        plt.plot(steps, mean_entropy_per_step, marker='s', linestyle='-', color='r', label='Entropy')
        plt.fill_between(steps,
                         mean_entropy_per_step - std_entropy_per_step,
                         mean_entropy_per_step + std_entropy_per_step,
                         color='r', alpha=0.2)

        plt.xlabel('Dialogue Step (intra_dialogue_id)')
        plt.ylabel('Mean Value')
        plt.title('Mean Cross-Entropy and Entropy per Dialogue Step')
        plt.grid(True)
        plt.xticks(steps)
        plt.legend()

        plt.tight_layout()

        plt.savefig('data/results/CE_and_entropy_' + filename + '.png')


def calculate_cross_entropy(distribution, target, scores_dict):
    """
    Calculate the cross-entropy given the predicted probabilities and the target item.
    :param distribution: A list of probabilities predicted by the model for the items.
    :param target: The target item which is the correct item.
    :param scores_dict: The dictionary containing the item scores.
    :return: The cross-entropy.
    """
    try:
        target_index = list(scores_dict.keys()).index(target)
    except ValueError:
        print(f"Warning: Target {target} not found in scores_dict")
        return np.nan

    true_distribution = np.zeros(len(distribution))
    true_distribution[target_index] = 1

    cross_entropy = log_loss([true_distribution], [distribution])
    print(true_distribution, distribution, sum(distribution))
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