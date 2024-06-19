import os
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

from analysis.item_probability_distr import compute_item_probability
from utils import group_dialogues_by_id, dump_sbs_row, load_game_dialogues, open_dump_sbs_file, open_game_sets, qa_to_str

# Configuration
load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
samples_number = 5
data_path = "./src/data/%s/8_mcrae"

dump_path = f"{data_path}/dialogues(gpt4o)_app1_onllama3_k5.csv" % "generation"

def generate_sbs_data(dialogues, dump_row, samples=5):
  print(samples)
  history = []

  with tqdm(total=len(dialogues), desc="SBS Analysis", unit="item") as pbar:
    for dial in dialogues:
      
      dialogue_id = dial['id']
      candidates = dial['candidates']
        
      for intra_dial in dial["intra_dialogue"]:
        
        pbar.set_description(f"SBS Analysis - Dialogue:{dialogue_id}.{intra_dial['id']}")
        
        question = intra_dial['question']
        answer = intra_dial['answer']
        history.append(qa_to_str(question, answer))
          
        results = compute_item_probability(history, candidates, samples, model_name='ollama-llama3')
      
        dump_row(
          dialogue_id,
          intra_dial["id"],
          dial["target"],
          question,
          answer,
          results["scores"],
          results["normalized_scores"],
          results["explanations"]
        )
      
      history = []
      pbar.update(1)



if __name__ == "__main__":
  parser=argparse.ArgumentParser()
  
  parser.add_argument("--start_dialogue_id", type=int)
  parser.add_argument("--end_dialogue_id", type=int)
  
  args=parser.parse_args()
  
  start_dialogue_id = -1
  end_dialogue_id = -1
  
  if(args.start_dialogue_id):
    start_dialogue_id = args.start_dialogue_id
    
  if(args.end_dialogue_id and args.end_dialogue_id > start_dialogue_id):
    end_dialogue_id = args.end_dialogue_id
   
    
  # Loading game sets
  games_sets = open_game_sets(f"{data_path}/contrast_sets.json" % "game_sets")
  
  last_dumped_dialogue_id , last_dumped_intra_dialogue_id = open_dump_sbs_file(dump_path)
    
  start_dialogue_id = last_dumped_dialogue_id if start_dialogue_id == -1 else start_dialogue_id

  # start_dialogue_id = 0
  # end_dialogue_id = -1

  # Opening dumped dialogues data
  raw_dumped_dialogues = load_game_dialogues(
    f"{data_path}/gpt-dialogues-gpt4.csv" % "generation",
    start_dialogue_id,
    end_dialogue_id
  )
  # Grouping rows by dialogue_id
  dialogues = group_dialogues_by_id(raw_dumped_dialogues, games_sets, start_dialogue_id)
  print(len(dialogues))
  generate_sbs_data(dialogues, dump_sbs_row(dump_path), samples=samples_number)