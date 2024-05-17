import csv
import json

def has_resurrected_items(dialogue_entropies):
  for i in range(0, len(dialogue_entropies) - 1 ):
      if(dialogue_entropies[i] < dialogue_entropies[i+1]):
        return True

def group_entropies_by_dialogue_id(reader: csv.DictReader):
  
  # Grouping lines by dialogue_id
  current_dialogue_id = -1
  max_length = 0
  entropies = {}
  
  for row in reader:
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

    
# Grouping dialogues rows by dialogue_id
def group_dialogues_by_id(dumped_dialogues, games_sets, current_dialogue_id):
  dialogues = []
  intra_dialogue = []

  for row in dumped_dialogues:
    dialogue_id = row['dialogue_id']

    if (int(dialogue_id) == int(current_dialogue_id)):
      current_target = row["target"]

    if(int(dialogue_id) != int(current_dialogue_id)):
      dialogues.append({
        "id" :  str(current_dialogue_id), 
        "intra_dialogue" : intra_dialogue,
        "target" : current_target,
        "candidates" : games_sets[str(current_dialogue_id)]['items']
      })
      current_dialogue_id = dialogue_id
      
      intra_dialogue = []

    # Reading each <question, answer> 
    question = row['question']
    answer = row['answer']
    
    intra_dialogue.append({
      "id" : row["intra_dialogue_id"],
      "question" : question,
      "answer" : answer,
    })
    
  return dialogues
      
def open_dump_sbs_file(filepath):
  last_intra_dialogue_id = -1
  last_dialogue = 0
  try:
    with open(filepath, 'r', newline='') as f:
      lines = f.readlines()
      if(len(lines) != 1):
        row = lines[-1].split(",")
        last_dialogue = int(row[0])
        last_intra_dialogue_id = int(row[1])
  except FileNotFoundError:
    with open(filepath, 'w', newline='') as f:
      write = csv.writer(f)
      write.writerow([
        "dialogue_id", # enumeration of games dialogues
        "intra_dialogue_id", # question/answer index inside dialogue with id = dialogue_id
        "target", # item assigned to user
        "question", # question made by the guesser
        "answer",
        "candidates_scores",
        "p_distribuition",
        "explanations"
      ])
      
  return last_dialogue, last_intra_dialogue_id

# HOF for dumping sbs analysis rows
def dump_sbs_row(filepath,):
  def dump_fn(dialogue_id, intra_dialogue_id, target, question, answer, candidate_scores, p_distribuition, explanations):
    with open(filepath, 'a', newline='') as f:
      write = csv.writer(f)
      write.writerow([
        dialogue_id, 
        intra_dialogue_id, 
        target, 
        question, 
        answer,
        candidate_scores,
        p_distribuition,
        explanations
      ])
      
  return dump_fn


def load_game_dialogues(filepath, first_dialogue_id, last_dialogue_id=-1):
  rows = []
  
  with open(filepath, newline='') as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
      dialogue_id = int(row["dialogue_id"])
      if( dialogue_id >= first_dialogue_id ):
        if(last_dialogue_id == -1 or dialogue_id <= last_dialogue_id):
          rows.append(row)
      
  return rows

    
def qa_to_str(question, answer):
  return f"- Question: {question} Answer: {answer}"
    
    
def open_game_sets(filepath):
  with open(filepath) as f:
    return json.load(f)
  
# Grouping sbs distr by dialogue_id
def group_sbs_data_by_dialogue_id(reader: csv.DictReader):
  
  dialogues = []
  dialogue_steps = []
  
  current_dialogue_id = 0
  for row in reader:
    dialogue_id = row["dialogue_id"]
    
    if(dialogue_id != ''):
      dialogue_id = int(dialogue_id)
      intra_dialogue_id = int(row["intra_dialogue_id"])
      
      if(dialogue_id != current_dialogue_id):
        dialogues.append({
          "dialogue_id" : current_dialogue_id,
          "intra_dialogues" : dialogue_steps
        })
        
        current_dialogue_id = dialogue_id
        dialogue_steps = []
      
      dialogue_steps.append({
        "intra_dialogue_id" : intra_dialogue_id,
        "p_distribuition" : json.loads(row["p_distribuition"].replace('\'', '"'))
      })
    
  return dialogues