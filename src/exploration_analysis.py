# building a function evaluating if gpt-q is exploiting (HS) or exploring (CS)
# compute the global information gain of each strategy (entropy differential dH)

import pandas as pd
import json
import matplotlib.pyplot as plt

# Load the dialogues data
#file_path_gpt3 = 'data/generation/8_mcrae/dialogues.csv'
file_path_gpt3 = 'data/generation/8_mcrae/dialogues(gpt3)_app1_on3_k5.csv'
dialogues_gpt3_df = pd.read_csv(file_path_gpt3)
#file_path_gpt4 = 'data/generation/8_mcrae/gpt-dialogues-gpt4.csv'
#file_path_gpt4 = 'data/generation/8_mcrae/dialogues(gpt4o)_sbs_k_five_gpt4o.csv'
file_path_gpt4 = 'data/generation/8_mcrae/dialogues_sbs_k_five_gpt4o.csv'
dialogues_gpt4_df = pd.read_csv(file_path_gpt4)

# Add a column to differentiate between the models
dialogues_gpt3_df['model'] = 'gpt3'
dialogues_gpt4_df['model'] = 'gpt4'

# Combine the dataframes
dialogues_df = pd.concat([dialogues_gpt3_df, dialogues_gpt4_df], ignore_index=True)

# Load the JSON file with possible items for each dialogue
json_file_path = 'data/game_sets/8_mcrae/contrast_sets.json'
with open(json_file_path, 'r') as file:
    contrast_sets = json.load(file)

# Extract the possible items for each dialogue
dialogue_items = {int(dialogue_id): data['items'] for dialogue_id, data in contrast_sets.items()}

# Function to classify strategies based on the presence of possible items in the question
def classify_strategy(question, possible_items):
    for item in possible_items:
        if item.lower() in question.lower():
            return "exploiting"
    return "exploring"

# Apply the classification function to all questions
dialogues_df['strategy'] = dialogues_df.apply(lambda row: classify_strategy(row['question'], dialogue_items[row['dialogue_id']]), axis=1)

# Calculate the percentages for each model
strategy_counts = dialogues_df.groupby(['model', 'strategy']).size().unstack(fill_value=0)
strategy_percentages = strategy_counts.div(strategy_counts.sum(axis=1), axis=0) * 100

# Display the strategy percentages
print(strategy_percentages)
# Set a different backend for Matplotlib
plt.switch_backend('agg')
# Plot the results with improved resolution and layout adjustments
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
strategy_percentages.plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Percentage')
ax.set_title('Exploring vs Exploiting Strategies by Model')
ax.set_xlabel('Model')
plt.xticks(rotation=0, ha='center')

# Save the plot to a file with increased resolution
plt.savefig('data/results/exploring_vs_exploiting_strategies_high_res.png', bbox_inches='tight')

# Inform the user about the saved plot
print("The plot has been saved as 'exploring_vs_exploiting_strategies_high_res.png' in the data/results directory.")

# Create a new column to represent the phase in the dialogue
dialogues_df['phase'] = dialogues_df['intra_dialogue_id']

# Calculate the maximum phase to ensure we cover all steps
max_phase = dialogues_df['phase'].max()

# Calculate the percentages of strategies used in each phase for each model, ensuring all phases are included
phase_strategy_counts = dialogues_df.groupby(['model', 'phase', 'strategy']).size().unstack(fill_value=0).reindex(pd.MultiIndex.from_product([dialogues_df['model'].unique(), range(max_phase + 1)]), fill_value=0)
phase_strategy_percentages = phase_strategy_counts.div(phase_strategy_counts.sum(axis=1), axis=0) * 100

# Plot the results with improved resolution and layout adjustments
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), dpi=300, sharex=True)

# Plot for GPT-3
phase_strategy_percentages.loc['gpt3'].plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_ylabel('Percentage')
axes[0].set_title('GPT-3: Exploring vs Exploiting Strategies by Phase')
axes[0].legend(loc='upper right')

# Plot for GPT-4
phase_strategy_percentages.loc['gpt4'].plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_ylabel('Percentage')
axes[1].set_title('GPT-4: Exploring vs Exploiting Strategies by Phase')
axes[1].legend(loc='upper right')

# Set common labels and titles
plt.xlabel('Phase of Dialogue')
plt.xticks(rotation=0, ha='center')

# Save the plot to a file with increased resolution
plt.savefig('data/results/exploring_vs_exploiting_strategies_by_phase_high_res.png', bbox_inches='tight')

# Inform the user about the saved plot
print("The plot showing the strategy usage by phase has been saved as"
      "'exploring_vs_exploiting_strategies_by_phase_high_res.png' in the data/results directory.")

'''
# Filter the DataFrame to include only rows for GPT-3 from the 12th step of the dialogue onwards
gpt3_late_steps_df = dialogues_df[(dialogues_df['model'] == 'gpt3') & (dialogues_df['phase'] >= 13)]

# Print the filtered DataFrame
print(gpt3_late_steps_df)

# Calculate statistics for exploration and exploitation strategies
strategy_counts_late_steps = gpt3_late_steps_df['strategy'].value_counts()
strategy_percentages_late_steps = strategy_counts_late_steps / strategy_counts_late_steps.sum() * 100

# Print the statistics
print("\nStrategy counts from 14th step onwards for GPT-3:")
print(strategy_counts_late_steps)
print("\nStrategy percentages from 12th step onwards for GPT-3:")
print(strategy_percentages_late_steps) '''

# Load entropy data from the provided CSV files
entropy_gpt3_app1_path = 'data/generation/8_mcrae/sbs_entropy_k_five_cleaned.csv'
entropy_gpt3_app2_path = 'data/generation/8_mcrae/sbs_entropy(gpt3)_k_five_gpt3_sim_app_apocalypse_cleaned.csv'
entropy_gpt4_app1_path = 'data/generation/8_mcrae/sbs_entropy(gpt4o)_k_five_gpt4o_cleaned.csv'
entropy_gpt4_app2_path = 'data/generation/8_mcrae/sbs_entropy(gpt4o)_k_five_gpt4o_sim_app.csv'
entropy_gpt3_app1_df = pd.read_csv(entropy_gpt3_app1_path)
entropy_gpt3_app2_df = pd.read_csv(entropy_gpt3_app2_path)
entropy_gpt4_app1_df = pd.read_csv(entropy_gpt4_app1_path)
entropy_gpt4_app2_df = pd.read_csv(entropy_gpt4_app2_path)

# Merge entropy data with dialogues using 'dialogue_id' and 'intra_dialogue_id' for GPT-3
dialogues_gpt3 = dialogues_df[dialogues_df['model'] == 'gpt3']
dialogues_gpt3 = dialogues_gpt3.merge(entropy_gpt3_app1_df, on=['dialogue_id', 'intra_dialogue_id'], suffixes=("", "_app1_gpt3"))
dialogues_gpt3 = dialogues_gpt3.merge(entropy_gpt3_app2_df, on=['dialogue_id', 'intra_dialogue_id'], suffixes=("", "_app2_gpt3"))
dialogues_gpt3 = dialogues_gpt3.rename(columns={'entropy': 'entropy_app1_gpt3'})

# Merge entropy data with dialogues using 'dialogue_id' and 'intra_dialogue_id' for GPT-4
dialogues_gpt4 = dialogues_df[dialogues_df['model'] == 'gpt4']
dialogues_gpt4 = dialogues_gpt4.merge(entropy_gpt4_app1_df, on=['dialogue_id', 'intra_dialogue_id'], suffixes=("", "_app1_gpt4"))
dialogues_gpt4 = dialogues_gpt4.merge(entropy_gpt4_app2_df, on=['dialogue_id', 'intra_dialogue_id'], suffixes=("", "_app2_gpt4"))
dialogues_gpt4 = dialogues_gpt4.rename(columns={'entropy': 'entropy_app1_gpt4'})

# Combine the two dataframes back into one
dialogues_entropy_df = pd.concat([dialogues_gpt3, dialogues_gpt4], ignore_index=True)

# Calculate the impact of each strategy on entropy reduction
strategy_impact_gpt3 = dialogues_gpt3.groupby(['strategy'])[['entropy_app1_gpt3', 'entropy_app2_gpt3']].mean()
strategy_impact_gpt4 = dialogues_gpt4.groupby(['strategy'])[['entropy_app1_gpt4', 'entropy_app2_gpt4']].mean()

# Rename the columns for better understanding in the plot
strategy_impact_gpt3 = strategy_impact_gpt3.rename(columns={'entropy_app1_gpt3': 'singular approach', 'entropy_app2_gpt3': 'simultaneous approach'})
strategy_impact_gpt4 = strategy_impact_gpt4.rename(columns={'entropy_app1_gpt4': 'singular approach', 'entropy_app2_gpt4': 'simultaneous approach'})

# Create plots to visualize the impact of strategies on entropy reduction

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot for GPT-3
strategy_impact_gpt3.plot(kind='bar', ax=axes[0], title='Impact of strategies on entropy for GPT-3 (GPT-3 dialogues)')
axes[0].set_xlabel('Strategy')
axes[0].set_ylabel('Entropy')

# Plot for GPT-4
strategy_impact_gpt4.plot(kind='bar', ax=axes[1], title='Impact of strategies on entropy for GPT-4 (GPT-4 dialogues))')
axes[1].set_xlabel('Strategy')
axes[1].set_ylabel('Entropy')

plt.tight_layout()

plt.tight_layout()
plt.savefig('data/results/impact_of_strategies_on_entropy.png')


# Calculate the differential entropy reduction for each strategy
def calculate_entropy_reduction(df, entropy_col, strategy_col):
    df[f'{strategy_col}_reduction'] = df.apply(
        lambda row: row[entropy_col] - df.loc[row.name + 1, entropy_col]
        if (row.name + 1 in df.index and row['phase'] < df.loc[row.name + 1, 'phase'])
        else None,
        axis=1
    )

    return df

# Apply the function to calculate entropy reduction for each model and strategy
dialogues_gpt3 = calculate_entropy_reduction(dialogues_gpt3, 'entropy_app1_gpt3', 'singular approach')
dialogues_gpt3 = calculate_entropy_reduction(dialogues_gpt3, 'entropy_app2_gpt3', 'simultaneous approach')
dialogues_gpt4 = calculate_entropy_reduction(dialogues_gpt4, 'entropy_app1_gpt4', 'singular approach')
dialogues_gpt4 = calculate_entropy_reduction(dialogues_gpt4, 'entropy_app2_gpt4', 'simultaneous approach')

# Calculate the mean entropy reduction for each strategy
strategy_impact_gpt3 = dialogues_gpt3.groupby(['strategy'])[['singular approach_reduction', 'simultaneous approach_reduction']].mean()
strategy_impact_gpt4 = dialogues_gpt4.groupby(['strategy'])[['singular approach_reduction', 'simultaneous approach_reduction']].mean()

# Rename the columns for better understanding in the plot
strategy_impact_gpt3 = strategy_impact_gpt3.rename(columns={'singular approach_reduction': 'singular approach', 'simultaneous approach_reduction': 'simultaneous approach'})
strategy_impact_gpt4 = strategy_impact_gpt4.rename(columns={'singular approach_reduction': 'singular approach', 'simultaneous approach_reduction': 'simultaneous approach'})

# Calcolo della varianza per ciascuna strategia e modello
variance_gpt3_app1 = dialogues_gpt3.groupby('strategy')['singular approach_reduction'].var()
variance_gpt4_app1 = dialogues_gpt4.groupby('strategy')['singular approach_reduction'].var()
variance_gpt3_app2 = dialogues_gpt3.groupby('strategy')['simultaneous approach_reduction'].var()
variance_gpt4_app2 = dialogues_gpt4.groupby('strategy')['simultaneous approach_reduction'].var()

print("Var GPT-3 (singular):", variance_gpt3_app1)
print("Var GPT-4 (singular):", variance_gpt4_app1)
print("Var GPT-3 (simultaneous):", variance_gpt3_app2)
print("Var GPT-4 (simultaneous):", variance_gpt4_app2)

# Create plots to visualize the impact of strategies on entropy reduction

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot for GPT-3
strategy_impact_gpt3.plot(kind='bar', ax=axes[0], title='Impact of Strategies on Entropy Reduction for GPT-3')
axes[0].set_xlabel('Strategy')
axes[0].set_ylabel('Entropy Reduction')

# Plot for GPT-4
strategy_impact_gpt4.plot(kind='bar', ax=axes[1], title='Impact of Strategies on Entropy Reduction for GPT-4')
axes[1].set_xlabel('Strategy')
axes[1].set_ylabel('Entropy Reduction')

plt.savefig('data/results/impact_of_strategies_on_entropy_reduction.png')

# Combine the mean entropy reduction for each strategy for GPT-3 and GPT-4 into a single dataframe for plotting
combined_strategy_impact = pd.concat([strategy_impact_gpt3, strategy_impact_gpt4], keys=['GPT-3', 'GPT-4']).reset_index()

# Create a plot to visualize the impact of strategies on entropy reduction with a uniform scale
fig, ax = plt.subplots(figsize=(12, 6))

# Plot for both GPT-3 and GPT-4
combined_strategy_impact.pivot(index='level_0', columns='strategy', values=['singular approach', 'simultaneous approach']).plot(kind='bar', ax=ax)

ax.set_title('Impact of strategies on Entropy reduction for GPT-3 and GPT-4')
ax.set_xlabel('Model and Strategy')
ax.set_ylabel('Mean Entropy Reduction H(t)-H(t+1)')
#ax.set_xticklabels([f"{row['level_0']} - {row['strategy']}" for _, row in combined_strategy_impact.iterrows()], rotation=45)
ax.legend(title='Approach')

plt.tight_layout()

plt.savefig('data/results/impact_of_strategies_on_entropy_reduction_2.png')
