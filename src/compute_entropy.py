import csv
import json
import math
import numpy as np

data_path = ".src/data/generation/8_mcrae/dialogues_sbs_k_five_gpt4o.csv"
filename = "sbs_entropy_k_five_gpt4o_cleaned"

to_clean = True
to_apocalypse = False

def main():
  
  rf = open(data_path, 'r', newline='')
  reader = csv.DictReader(rf, delimiter=",")
  zeros_list = np.zeros(8)

  with open(f".src/data/generation/8_mcrae/{filename}.csv", "w", newline='') as df:
    csv.writer(df).writerow([
      "dialogue_id",
      "intra_dialogue_id",
      "entropy"
    ])

  previous_scores= []
  for row in reader:
      raw_scores = row["candidates_scores"].replace('\'', '"')
      raw_distribuition = row["p_distribuition"].replace('\'', '"')
      
      if len(raw_scores) != 0:
        
        json_scores = json.loads(raw_scores)
        json_distribuition = json.loads(raw_distribuition)
        
        scores = list(json_scores.values())
        distr = list(json_distribuition.values())
        
        if(row["intra_dialogue_id"] == "0"):
          previous_scores = scores
          previous_distr = distr
        
        if(to_clean):
          distr, scores = clean_scores(scores, previous_scores)
          previous_scores = scores
        
        entropy = 0
        if np.array_equal(zeros_list, distr):
          if to_apocalypse:
             # set entropy to a invalid value to be able to filter it later
            entropy = 1.0
          else:
            # otherwise, set entropy to max value
            entropy = -3.0
        for c in distr:
          if(c != 0):
            entropy = entropy + c * math.log(c, 2)
        entropy = round(-1 * entropy, 4)
        print(entropy)
      else:
        entropy = ''
        
      with open(f".src/data/generation/8_mcrae/{filename}.csv", "a", newline='') as df:
        csv.writer(df).writerow([
          row["dialogue_id"],
          row["intra_dialogue_id"],
          entropy
        ])
  rf.close()
      
      
# This functions takes as argument the previous (normalized) scores and the current ones readen from the csv
# and excludes the current candidates (from the current scores) which have been excluded in the previous steps
def clean_scores(current_scores, previous_scores, samples=5):
  
  cleaned_scores = []
  
  for i, ith_score in enumerate(current_scores):
    if(previous_scores[i] == 0):
      cleaned_scores.append(0)
    else:
      cleaned_scores.append(ith_score)

  scores_sum = 0
  for ith_score in cleaned_scores:
    scores_sum += ith_score
    
  cleaned_distribuition = []
  for ith_score in cleaned_scores:
    try:
      normalized_ith_score = round(ith_score / scores_sum, 4)
    except ZeroDivisionError :
      normalized_ith_score = 0
    cleaned_distribuition.append(normalized_ith_score)
    
  return cleaned_distribuition, cleaned_scores
  
main()