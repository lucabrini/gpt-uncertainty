
import csv
import numpy

data_path = "./src/data/generation/8_mcrae/sbs_entropy.csv"
filename = "entropy_statistics_from_dialogue_end"

def main():
  dialogues_entropies, max_dialogue_length = group_by_dialogue_id()
  
  entropies_by_distances = []
  
  for _ in range(0, max_dialogue_length, 1):
    entropies_by_distances.append([])
  
  for dialogue_entropy in dialogues_entropies.values():
    dialogue_length = len(dialogue_entropy)
    for step_index, step_entropy in enumerate(dialogue_entropy):
      distance_from_end = dialogue_length - step_index - 1
      
      entropies_by_distances[distance_from_end].append(step_entropy)
  
  with open(f"./src/data/generation/8_mcrae/{filename}.csv", "w", newline='') as df:
    csv.writer(df).writerow([
      "distance",
      "mean",
      "std"
    ])
  
  for (distance, entropies_by_distance) in enumerate(entropies_by_distances):
    std = numpy.std(entropies_by_distance)
    mean = numpy.mean(entropies_by_distance)
    print(distance, std, mean)
    
    with open(f"./src/data/generation/8_mcrae/{filename}.csv", "a", newline='') as df:
      csv.writer(df).writerow([
       distance, std, mean
      ])
    
  
def group_by_dialogue_id():
  
  rf = open(data_path, 'r', newline='')
  reader = csv.DictReader(rf, delimiter=",")
  
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

main()