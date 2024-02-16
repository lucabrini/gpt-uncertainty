import os
import csv
import json
from dotenv import load_dotenv
from string import Template
from tqdm import tqdm
import openai


def main():
  load_dotenv()
  openai_api_key = os.environ.get("OPENAI_API_KEY")
  data_path = "./src/data/%s/8_mcrae"

  # Loading games sets. We need the candidates of each game set
  games_sets = open_game_sets(data_path % "game_sets")

  current_dialogue_id = open_step_by_step_csv(data_path % "generation")
  print(current_dialogue_id)
  # Opening dumped dialogues data
  dumped_dialogues = open_dialogues(data_path % "generation", current_dialogue_id)

  history = []
  candidates = []

  for row in tqdm(dumped_dialogues, desc="Step by step analysis", unit="item"):

    dialogue_id = row['dialogue_id']

    # If we have a new dialogue, reset variables
    if(dialogue_id != current_dialogue_id):
      current_dialogue_id = dialogue_id
      history = []
      candidates = games_sets[current_dialogue_id]['items']

    # Reading each <question, answer>
    question = row['question']
    answer = row['answer']


    # if("correct" not in answer): # Excluding last dialogue question / answer
    history.append(
      f"-{question} {answer}"
    )

    remaining_candidates = list_remaining_candidates(history, candidates)
    #candidates = remaining_candidates

    # Computing probability
    p = 1 / len(remaining_candidates)

    dump_row(
      data_path % "generation",
      dialogue_id,
      row["intra_dialogue_id"],
      row["target"],
      question,
      answer,
      remaining_candidates,
      p
    )


def build_prompt(candidates, dialogue_history):
  template = Template((
    "You will be given of a dialogue of the 20 questions game. "
    "You have to list out absolutely all the items from the given candidates set that satisfy each <question, answer> in the given dialogue."
    "\n\n"
    "The output should strictly use the following template: \n"
    "EXPLANATION: [insert your analysis of each candidated items];"
    "CANDIDATES: item1, item2, item3"
    "\n\n"
    "Candidates: $candidates.\n"
    "Dialogue: \n"
    "$dialogue_history"
  ))

  return template.substitute(
    candidates=candidates,
    dialogue_history=dialogue_history
  )

def open_step_by_step_csv(data_path):
  with open(f"{data_path}/dialogues_step_by_step.csv", 'r', newline='') as f:
    lines = f.readlines()
    if(len(lines) == 0):
      write = csv.writer(f)
      write.writerow([
        "dialogue_id", # enumeration of games dialogues
        "intra_dialogue_id", # question/answer index inside dialogue with id = dialogue_id
        "target", # item assigned to user
        "question", # question made by the guesser
        "answer",
        "candidates",
        "probability",
        "explanation"
      ])

      last_dialogue = -1
    else:
      print(lines[-1].split(",")[0])
      last_dialogue = int(lines[-1].split(",")[0])

  return last_dialogue

def open_game_sets(data_path):
  with open(f"{data_path}/contrast_sets.json") as f:
    return json.load(f)

def open_dialogues(data_path, first_dialogue_id):
  rows = []
  with open(f"{data_path}/dialogues.csv", newline='') as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
      if(int(row["dialogue_id"]) >= first_dialogue_id):
        rows.append(row)

  return rows

def list_remaining_candidates(history, candidates):
  history_str = "\n".join(history)
  candidates_str = ", ".join(candidates)
  prompt = build_prompt(candidates_str, history_str)

  # Asking gpt to list out the remaining items
  response = openai.chat.completions.create(
    temperature=0,
    model='gpt-3.5-turbo',
    messages=[{'role': "system", 'content': prompt}],
  ).choices[0].message.content

  candidates = response.split("CANDIDATES: ")[1].split(", ")
  explanation = response.split("EXPLANATION: ")[1]

  return candidates


def dump_row(data_path, dialogue_id, intra_dialogue_id, target, question, answer, candidates, probability):
  with open(f"{data_path}/dialogues_step_by_step.csv", 'a', newline='') as f:
    write = csv.writer(f)
    write.writerow([
      dialogue_id,
      intra_dialogue_id,
      target,
      question,
      answer,
      candidates,
      probability
    ])


if __name__ == "__main__":
  main()
