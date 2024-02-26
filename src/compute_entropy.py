import csv
import json
import math

data_path = "./src/data/generation/8_mcrae/dialogues_step_by_step_distr.csv"
filename = "sbs_entropy"

def main():
  
  rf = open(data_path, 'r', newline='')
  reader = csv.DictReader(rf, delimiter=",")

  with open(f"./src/data/generation/8_mcrae/{filename}.csv", "w", newline='') as df:
    csv.writer(df).writerow([
      "dialogue_id",
      "intra_dialogue_id",
      "entropy"
    ])

  previous_distribuition = []
  for row in reader:
      raw_distribuition = row["p_distribuition"].replace('\'', '"')
      
      if len(raw_distribuition) != 0:
        
        json_distribuition = json.loads(raw_distribuition)
        distribuition = list(json_distribuition.values())
        
        if(row["intra_dialogue_id"] == "0"):
          previous_distribuition = distribuition
        
        #distribuition = exclude_candidates(distribuition, previous_distribuition)
        previous_distribuition = distribuition
        
        entropy = 0
        for c in distribuition:
          if(c != 0):
            entropy = entropy + c * math.log(c, 2)
        entropy = round(-1 * entropy, 4)
        print(entropy)
      else:
        entropy = ''
        
      with open(f"./src/data/generation/8_mcrae/{filename}.csv", "a", newline='') as df:
        csv.writer(df).writerow([
          row["dialogue_id"],
          row["intra_dialogue_id"],
          entropy
        ])
  rf.close()
      
def exclude_candidates(current_distribuition, previous_distribuition):
  
  correct_distribuition = []
  
  for i in range(0, len(current_distribuition), 1):
    if(previous_distribuition[i] == 0):
      correct_distribuition.append(0)
    else:
      correct_distribuition.append(current_distribuition[i])
      
  print(previous_distribuition)
  print(current_distribuition)
      
  return correct_distribuition

main()