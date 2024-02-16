import os
import csv
import json
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

from analysis.item_probability_distr import compute_item_probability

# Configuration
load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
samples_number = 5
data_path = "./src/data/%s/8_mcrae"

def main(start_dialogue_id, last_dialogue_id):
  
  # Loading games sets. We need the candidates of each game set
  games_sets = open_game_sets(data_path % "game_sets")
  
  last_dumped_dialogue_id , last_dumped_intra_dialogue_id = open_dump_file(data_path % "generation")
  print(last_dumped_dialogue_id, last_dumped_intra_dialogue_id)
  
  start_dialogue_id = last_dumped_dialogue_id if start_dialogue_id == -1 else start_dialogue_id
  # Opening dumped dialogues data
  raw_dumped_dialogues = load_game_dialogues(
    data_path % "generation",
    start_dialogue_id,
    last_dialogue_id
  )
  
  history = []
    
  # Grouping rows by dialogue_id
  dialogues = build_dialogues(raw_dumped_dialogues, start_dialogue_id)

  for dial in tqdm(dialogues, desc="Step by step analysis", unit="item"):
    
    dialogue_id = dial['id']
    candidates = games_sets[dialogue_id]['items']
      
    for intra_dial in dial["intra_dialogue"]:
      question = intra_dial['question']
      answer = intra_dial['answer']
      history.append(qa_to_str(question, answer))
        
      results = compute_item_probability(history, candidates, openai_api_key)
    
      dump_row(
        data_path % "generation",
        dialogue_id,
        intra_dial["id"],
        dial["target"],
        question,
        answer,
        results["scores"],
        results["normalized_scores"],
      )
    
    history = []
    
# Grouping rows by dialogue_id
def build_dialogues(dumped_dialogues, current_dialogue_id):
  dialogues = []
  intra_dialogue = []

  for row in dumped_dialogues:
    dialogue_id = row['dialogue_id']
    
    if(int(dialogue_id) != current_dialogue_id):
      dialogues.append({
        "id" :  str(current_dialogue_id), 
        "intra_dialogue" : intra_dialogue ,
        "target" : row["target"]
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
      
      
def qa_to_str(question, answer):
  return f"- Question: {question} Answer: {answer}"


def open_dump_file(data_path):
  last_intra_dialogue_id = -1
  last_dialogue = 0
  try:
    with open(f"{data_path}/dialogues_step_by_step_distr.csv", 'r', newline='') as f:
      lines = f.readlines()
      if(len(lines) != 1):
        row = lines[-1].split(",")
        last_dialogue = int(row[0])
        last_intra_dialogue_id = int(row[1])
  except FileNotFoundError:
    with open(f"{data_path}/dialogues_step_by_step_distr.csv", 'w', newline='') as f:
      write = csv.writer(f)
      write.writerow([
        "dialogue_id", # enumeration of games dialogues
        "intra_dialogue_id", # question/answer index inside dialogue with id = dialogue_id
        "target", # item assigned to user
        "question", # question made by the guesser
        "answer",
        "candidates_scores"
        "p_distribuition",
      ])
      
  return last_dialogue, last_intra_dialogue_id
    
    
def open_game_sets(data_path):
  with open(f"{data_path}/contrast_sets.json") as f:
    return json.load(f)
  
  
def load_game_dialogues(data_path, first_dialogue_id, last_dialogue_id=-1):
  rows = []
  
  with open(f"{data_path}/dialogues.csv", newline='') as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
      dialogue_id = int(row["dialogue_id"])
      if( dialogue_id >= first_dialogue_id ):
        if(last_dialogue_id == -1 or dialogue_id <= last_dialogue_id):
          rows.append(row)
      
  return rows


def dump_row(data_path, dialogue_id, intra_dialogue_id, target, question, answer, candidate_scores, p_distribuition):
  with open(f"{data_path}/dialogues_step_by_step_distr.csv", 'a', newline='') as f:
    write = csv.writer(f)
    write.writerow([
      dialogue_id, 
      intra_dialogue_id, 
      target, 
      question, 
      answer,
      candidate_scores,
      p_distribuition
    ])


if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  
  parser.add_argument("--start_dialogue_id")
  parser.add_argument("--last_dialogue_id")
  
  args=parser.parse_args()
  
  start_dialogue_id = -1
  last_dialogue_id = -1
  
  if(args.start_dialogue_id):
    start_dialogue_id = args.start_dialogue_id
    
  if(args.last_dialogue_id and args.last_dialogue_id > start_dialogue_id):
    last_dialogue_id = args.last_dialogue_id
  
  main(
    start_dialogue_id=start_dialogue_id,
    last_dialogue_id=last_dialogue_id,
  )