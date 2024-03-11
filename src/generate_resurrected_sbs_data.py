
import csv
from generate_sbs_data import generate_sbs_data

from utils import build_sbs_dialogues, dump_sbs_row, group_entropies_by_dialogue_id, load_game_dialogues, open_dump_sbs_file, open_game_sets

data_path = "./src/data/%s/8_mcrae"
sbs_path = f"{data_path}/sbs_entropy.csv" % "generation"
dump_path = f"{data_path}/dialogues_step_by_step_resurr_distr.csv" % "generation"

def main():
  
  # Read the step by step analisys and find the rows with resurrected items.
  # This happens when entropy of step t is bigger than the entropy of step t+1
  rf = open(sbs_path, 'r', newline='')
  reader = csv.DictReader(rf, delimiter=",")
  
  dialogues_entropies, max_dialogue_length = group_entropies_by_dialogue_id(reader)
  print(dialogues_entropies)
  
  resurrected_dialogues = []
  
  for dialogue_id in dialogues_entropies:
    is_resurrected = False
    dialogue_entropies = dialogues_entropies[dialogue_id]
    
    for i in range(0, len(dialogue_entropies) - 1 ):
      if(dialogue_entropies[i] < dialogue_entropies[i+1] and not is_resurrected):
        is_resurrected = True
        resurrected_dialogues.append(dialogue_id)
        
  print(resurrected_dialogues)
        
  games_sets = open_game_sets(f"{data_path}/contrast_sets.json" % "game_sets")
  last_dumped_dialogue_id , last_dumped_intra_dialogue_id = open_dump_sbs_file(dump_path)

  raw_dumped_dialogues = load_game_dialogues( f"{data_path}/dialogues.csv" % "generation", 0)
    
  # Grouping rows by dialogue_id
  all_dialogues = build_sbs_dialogues(raw_dumped_dialogues, games_sets, 0)
  
  # Filtering the dialogues with resurrected items
  dialogues = []
      
  generate_sbs_data(dialogues, dump_sbs_row(dump_path), 10)
  
main()