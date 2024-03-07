
import csv

from utils import group_by_dialogue_id


data_path = "./src/data/generation/8_mcrae/sbs_entropy.csv"
filename = ""

def main():
  
  # Read the step by step analisys and find the rows with resurrected items.
  # This happens when entropy of step t is bigger than the entropy of step t+1
  rf = open(data_path, 'r', newline='')
  reader = csv.DictReader(rf, delimiter=",")
  
  dialogues_entropies, max_dialogue_length = group_by_dialogue_id(reader)
  print(dialogues_entropies)
  
  resurrected_dialogues = []
  
  for dialogue_id in dialogues_entropies:
    is_resurrected = False
    dialogue_entropies = dialogues_entropies[dialogue_id]
    
    for i in range(0, len(dialogue_entropies) - 1 ):
      if(dialogue_entropies[i] < dialogue_entropies[i+1] and not is_resurrected):
        is_resurrected = True
        resurrected_dialogues.append(dialogue_id)
        
  print(len(resurrected_dialogues))
        
      
      
  
  
main()