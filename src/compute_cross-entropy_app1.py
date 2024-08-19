import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import matplotlib.patches as mpatches

GPT_3_ON_3 = False
GPT_3_ON_4 = False
GPT_4_ON_3 = False
GPT_4_ON_4 = True

if GPT_3_ON_3:
    plot_legend = "Median Cross-Entropy and Entropy per dialogue step (GPT-3 on GPT-3 dialogues, singular approach)"
    plot_name_end = "gpt3_on3_app1.png"
    data_path = "data/generation/8_mcrae/dialogues(gpt3)_app1_on3_k5.csv"
    filename = "data/generation/8_mcrae/sbs_entropy_k_five_cleaned.csv"

elif GPT_3_ON_4:
    plot_legend = "Median Cross-Entropy and Entropy per dialogue step (GPT-3 on GPT-4 dialogues, singular approach)"
    plot_name_end = "gpt3_on4_app1.png"
    data_path = "data/generation/8_mcrae/dialogues(gpt4o)_sbs_k_five_gpt3.csv"
    filename = "data/generation/8_mcrae/entropy_gpt4o_app1_on3_k5_apocalypse_cleaned.csv"

elif GPT_4_ON_3:
    plot_legend = "Median Cross-Entropy and Entropy per dialogue step (GPT-4 on GPT-3 dialogues, singular approach)"
    plot_name_end = "gpt4_on3_app1.png"
    data_path = "data/generation/8_mcrae/dialogues_sbs_k_five_gpt4o.csv"
    filename = "data/generation/8_mcrae/sbs_entropy_k_five_gpt4o_cleaned.csv"

elif GPT_4_ON_4:
    plot_legend = "Median Cross-Entropy and Entropy per dialogue step (GPT-4 on GPT-4 dialogues, singular approach)"
    plot_name_end = "gpt4_on4_app1.png"
    data_path = "data/generation/8_mcrae/dialogues(gpt4o)_sbs_k_five_gpt4o.csv"
    filename = "data/generation/8_mcrae/sbs_entropy(gpt4o)_k_five_gpt4o.csv"

to_clean = True
to_apocalypse = True

plot = True


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
    dialogue_id_column = []
    intra_dialogue_id_column = []

    previous_scores = []

    end_count = 0
    correct_count = 0
    correct_dialogues = []
    for row in reader:

        dialogue_id_column.append(row['dialogue_id'])
        intra_dialogue_id_column.append(row['intra_dialogue_id'])


        raw_scores = row["candidates_scores"].replace('\'', '"')
        raw_distribution = row["p_distribuition"].replace('\'', '"')
        target = row['target']
        question = row['question']
        if target in question:
            print("Correct! The item is ", target)
            correct_count += 1
        if len(row['answer'])>5:
            end_count +=1
            if target in question:
                correct_dialogues.append(row['dialogue_id'])

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
    df["dialogue_id"] = dialogue_id_column
    df["intra_dialogue_id"] = intra_dialogue_id_column
    rf.close()

    print("Number of correct guesses: ", correct_count, end_count)
    print("Percetage of correct guesses: ", correct_count/end_count * 100, "%")

    # Write the updated dataframe back to the file
    df = adjust_df(df)
    #df.to_csv(filename, index=False)

    # Check if the necessary columns exist in the DataFrame
    if 'entropy' in df.columns and 'cross_entropy' in df.columns:
        # Compute the correlation between 'entropy' and 'cross_entropy'
        correlation = df['entropy'].corr(df['cross_entropy'])
        print(f"The correlation between entropy and cross_entropy is: {correlation}")
    else:
        print("The DataFrame does not contain the required columns 'entropy' and 'cross_entropy'.")

    if plot:
        # Ensure a compatible backend is used
        matplotlib.use('Agg')

        # Group by 'intra_dialogue_id' and calculate the median and ranges for cross-entropy and entropy
        median_cross_entropy_per_step = df.groupby('intra_dialogue_id')['cross_entropy'].median()
        iqr_cross_entropy_per_step = df.groupby('intra_dialogue_id')['cross_entropy'].apply(
            lambda x: x.quantile(0.75) - x.quantile(0.25))
        median_entropy_per_step = df.groupby('intra_dialogue_id')['entropy'].median()
        iqr_entropy_per_step = df.groupby('intra_dialogue_id')['entropy'].apply(
            lambda x: x.quantile(0.75) - x.quantile(0.25))

        # Function to identify outliers using IQR
        def identify_outliers(data):
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
            return outliers

        # Identify outliers
        cross_entropy_outliers = df.groupby('intra_dialogue_id')['cross_entropy'].apply(identify_outliers)
        entropy_outliers = df.groupby('intra_dialogue_id')['entropy'].apply(identify_outliers)

        # Remove outliers for mean and std computation
        filtered_cross_entropy = df[~df.index.isin(cross_entropy_outliers.index)].groupby('intra_dialogue_id')[
            'cross_entropy']
        filtered_entropy = df[~df.index.isin(entropy_outliers.index)].groupby('intra_dialogue_id')['entropy']

        median_filtered_cross_entropy_per_step = filtered_cross_entropy.median()
        iqr_filtered_cross_entropy_per_step = filtered_cross_entropy.std()
        median_filtered_entropy_per_step = filtered_entropy.median()
        iqr_filtered_entropy_per_step = filtered_entropy.std()

        # Print mean and std values for debugging
        print("Median Cross-Entropy per step (filtered):", median_filtered_cross_entropy_per_step)
        print("IQR Cross-Entropy per step (filtered):", iqr_filtered_cross_entropy_per_step)
        print("Median Entropy per step (filtered):", median_filtered_entropy_per_step)
        print("IQR Entropy per step (filtered):", iqr_filtered_entropy_per_step)

        # Create a range for x-axis (dialogue steps)
        steps = range(len(median_filtered_cross_entropy_per_step))

        # Flatten the outliers for plotting
        cross_entropy_outliers_flat = cross_entropy_outliers.explode().reset_index()
        entropy_outliers_flat = entropy_outliers.explode().reset_index()

        # Plot the mean cross-entropy and entropy with standard deviation as confidence intervals
        plt.figure(figsize=(10, 6), dpi=300)

        # Plot median cross-entropy
        plt.plot(steps, median_filtered_cross_entropy_per_step, marker='o', linestyle='-', color='dodgerblue',
                 label='Cross-Entropy')
        plt.fill_between(steps,
                         median_filtered_cross_entropy_per_step - iqr_filtered_cross_entropy_per_step,
                         median_filtered_cross_entropy_per_step + iqr_filtered_cross_entropy_per_step,
                         color='dodgerblue', alpha=0.2)

        # Plot median entropy
        plt.plot(steps, median_filtered_entropy_per_step, marker='s', linestyle='-', color='crimson', label='Entropy')
        plt.fill_between(steps,
                         median_filtered_entropy_per_step - iqr_filtered_entropy_per_step,
                         median_filtered_entropy_per_step + iqr_filtered_entropy_per_step,
                         color='crimson', alpha=0.2)

        plt.fill_between(steps,
                         median_cross_entropy_per_step - iqr_cross_entropy_per_step / 2,
                         median_cross_entropy_per_step + iqr_cross_entropy_per_step / 2,
                         color='dodgerblue', alpha=0.2)

        plt.fill_between(steps,
                         median_entropy_per_step - iqr_entropy_per_step / 2,
                         median_entropy_per_step + iqr_entropy_per_step / 2,
                         color='crimson', alpha=0.2)

        # Plot outliers
        plt.scatter(cross_entropy_outliers_flat['intra_dialogue_id'], cross_entropy_outliers_flat['cross_entropy'],
                    color='dodgerblue', s=10)
        plt.scatter(entropy_outliers_flat['intra_dialogue_id'], entropy_outliers_flat['entropy'], color='crimson', s=10)

        plt.xlabel('dialogue step')
        plt.ylabel('median value')
        plt.title(plot_legend)
        plt.grid(True)
        plt.xticks(range(len(median_filtered_cross_entropy_per_step)))
        plt.legend()

        plt.tight_layout()

    plt.savefig('data/results/CE_'+ plot_name_end)

    plot_box_plots(df)

    # Box plot for entropy and cross-entropy
    plt.figure(figsize=(10, 6), dpi=300)

    # Data for box plot
    entropy_data = [df[df['intra_dialogue_id'] == i]['entropy'].dropna() for i in df['intra_dialogue_id'].unique()]
    cross_entropy_data = [df[df['intra_dialogue_id'] == i]['cross_entropy'].dropna() for i in
                          df['intra_dialogue_id'].unique()]

    # Create the box plot
    plt.boxplot(entropy_data, positions=np.arange(len(entropy_data)) - 0.2, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor="crimson"), showmeans=True, labels=np.arange(len(entropy_data)))
    plt.boxplot(cross_entropy_data, positions=np.arange(len(cross_entropy_data)) + 0.2, widths=0.4, patch_artist=True,
                boxprops=dict(facecolor="dodgerblue"), showmeans=True, labels=np.arange(len(cross_entropy_data)))

    # Add the title and labels
    plt.title('Box Plot of Entropy and Cross-Entropy per Dialogue Step (GPT-4 on GPT-3 dialogues, singular approach)')
    plt.xlabel('Dialogue Step')
    plt.ylabel('Value')
    plt.grid(True)
    # Create custom legend for box plots
    entropy_patch = mpatches.Patch(color='crimson', label='Entropy')
    cross_entropy_patch = mpatches.Patch(color='dodgerblue', label='Cross-Entropy')
    plt.legend(handles=[entropy_patch, cross_entropy_patch])

    plt.tight_layout()

    plt.savefig('data/results/box_plot_CE_' + plot_name_end)
    plt.show()
    plt.close()


def calculate_cross_entropy(distribution, target, scores_dict):
    """
    Calculate the cross-entropy given the predicted probabilities and the target item.
    :param distribution: A list of probabilities predicted by the model for the items.
    :param target: The target item which is the correct item.
    :param scores_dict: The dictionary containing the item scores.
    :return: The cross-entropy.
    """
    # Find the index of the target item
    print(list(scores_dict.keys()), target)
    target_index = list(scores_dict.keys()).index(target)
    true_distribution = np.zeros(len(distribution))
    true_distribution[target_index] = 1
    epsilon = 1/8.0
    predicted_distribution = np.clip(distribution, epsilon, 1)

    cross_entropy = -np.sum(true_distribution * np.log2(predicted_distribution))

    ### DEBUG, REMOVE
    '''
    true_distribution = [1,0,0,0,0,0,0,0]
    predicted_distribution = [0,0,0,0,0,0,0,1]
    epsilon = 0.01
    cross_entropy = log_loss(true_distribution, predicted_distribution)
    predicted_distribution = np.clip(predicted_distribution, epsilon, 1 - epsilon)
    my_cross_entropy = -np.sum(true_distribution * np.log(predicted_distribution))
    print(cross_entropy, my_cross_entropy)
    exit()
    '''

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


def adjust_df(df):
    # Find the maximum intra_dialogue_id
    df['dialogue_id'] = df['dialogue_id'].astype(int)
    df['intra_dialogue_id'] = df['intra_dialogue_id'].astype(int)
    max_intra_id = df['intra_dialogue_id'].max()

    # Lista per tenere traccia delle nuove righe da aggiungere
    new_rows = []

    # Iterate through each dialogue_id in the DataFrame
    for dialogue_id in df['dialogue_id'].unique():
        last_entropy = None
        last_cross_entropy = None

        # For each intra_dialogue_id in the range, check if it exists, else add a new row
        for intra_id in range(max_intra_id + 1):
            row_exists = ((df['dialogue_id'] == dialogue_id) & (df['intra_dialogue_id'] == intra_id)).any()

            if row_exists:
                # Update the last known values of entropy and cross_entropy
                last_entropy = df[(df['dialogue_id'] == dialogue_id) & (df['intra_dialogue_id'] == intra_id)]['entropy'].values[0]
                last_cross_entropy = df[(df['dialogue_id'] == dialogue_id) & (df['intra_dialogue_id'] == intra_id)]['cross_entropy'].values[0]
            else:
                if last_entropy is not None and last_cross_entropy is not None:
                    # Use the last known values for missing rows
                    new_rows.append({'dialogue_id': dialogue_id, 'intra_dialogue_id': intra_id,
                                     'entropy': last_entropy, 'cross_entropy': last_cross_entropy})

    # Creare un DataFrame con le nuove righe
    new_rows_df = pd.DataFrame(new_rows)

    # Concatenare il DataFrame originale con le nuove righe
    df = pd.concat([df, new_rows_df], ignore_index=True)

    # Sort the DataFrame by dialogue_id and intra_dialogue_id to maintain order
    df = df.sort_values(by=['dialogue_id', 'intra_dialogue_id']).reset_index(drop=True)

    return df

def plot_box_plots(df):
    # Preparare i dati per il box plot
    data = [df['entropy'].dropna(), df['cross_entropy'].dropna()]

    # Creare il box plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.boxplot(data, labels=['Entropy', 'Cross-Entropy'], patch_artist=True, showmeans=True)

    # Aggiungere il titolo e le etichette degli assi
    plt.title('Box Plot of Entropy and Cross-Entropy')
    plt.ylabel('Value')
    plt.grid(True)

    # Salvare il plot
    plt.savefig('data/results/box_plot_entropy_cross_entropy_gpt3_on3_app1.png')


main()