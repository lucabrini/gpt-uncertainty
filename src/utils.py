import csv


def group_by_dialogue_id(reader: csv.DictReader):
  
  # Grouping lines by dialogue_id
  current_dialogue_id = -1
  max_length = 0
  entropies = {}
  
  for row in reader:
    print(row)
    dialogue_id = row["dialogue_id"]
    if(dialogue_id != ''):
      dialogue_id = int(dialogue_id)
      intra_dialogue_id = int(row["intra_dialogue_id"])
      if(dialogue_id != current_dialogue_id):
        entropies[dialogue_id] = []
        current_dialogue_id = dialogue_id
      
      entropies[dialogue_id].append(float(row["entropy"]))    
      
      if(intra_dialogue_id > max_length):
        max_length = intra_dialogue_id
    
  return entropies, max_length + 1