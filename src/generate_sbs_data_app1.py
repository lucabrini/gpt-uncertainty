import os
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

from analysis.item_probability_distr import compute_item_probability
from utils import group_dialogues_by_id, dump_sbs_row, load_game_dialogues, open_dump_sbs_file, open_game_sets, qa_to_str

# Configuration
load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
samples_number = 20
model = "gpt4"
dialogues = "gpt4o"

dialogues_path = f"dialogues-{dialogues}.csv"
data_path = "./src/data/%s/8_mcrae"

if "gpt" in model:
  model_path = "gpt"
elif "llama" in model:
  model_path = "llama"

if "gpt" in dialogues:
  dialogues_sub_path = 'gpt'
elif "llama" in dialogues:
  dialogues_sub_path = 'llama'

dump_path = f"{data_path}/{model_path}/dialogues/app1_{model}on{dialogues}_k{samples_number}.csv" % "generation"

def generate_sbs_data(dialogues, dump_row, model_name, samples=5):
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
          
        results = compute_item_probability(history, candidates, openai_api_key=openai_api_key, 
                                           samples=samples, model_name=model_name)
      
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
    f"{data_path}/{dialogues_sub_path}/dialogues/{dialogues_path}" % "generation",
    start_dialogue_id,
    end_dialogue_id
  )
  # Grouping rows by dialogue_id
  dialogues = group_dialogues_by_id(raw_dumped_dialogues, games_sets, start_dialogue_id)

  print("Data path: ", dump_path)
  print("Dialogues path: ", dialogues_path)
  print("Dialogues length: ", len(dialogues))
  print("K sample: ", samples_number)
  print("Model: ", model)
  generate_sbs_data(dialogues, dump_sbs_row(dump_path), samples=samples_number, model_name=model)