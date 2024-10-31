import os
import re
import csv
import json
from dotenv import load_dotenv
from string import Template
from tqdm import tqdm
import openai
import ollama
from collections import defaultdict
import string


def main():
    load_dotenv()
    samples = 20
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

    dump_path = f"{data_path}/{model_path}/dialogues/app2_{model}on{dialogues}_k{samples}.csv" % "generation"
    dialogues_dump_path = f"{data_path}/{dialogues_sub_path}/dialogues/{dialogues_path}" % "generation"

    print("Data path: ", dump_path)
    print("Dialogues path: ", dialogues_path)
    print("K sample: ", samples)
    print("Model: ", model)

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Loading games sets. We need the candidates of each game set
    games_sets = open_game_sets(data_path % "game_sets")

    # Opening dumped dialogues data
    dumped_dialogues = open_dialogues(dialogues_dump_path)
    print("Dialogues length: ", len(dumped_dialogues))
    # print(dumped_dialogues)

    open_step_by_step_csv(dump_path)

    current_dialogue = -1
    history = []
    candidates = []

    for row in tqdm(dumped_dialogues, desc="Step by step analysis", unit="item"):

        dialogue_id = row['dialogue_id']

        # If we have a new dialogue, reset variables
        if (dialogue_id != current_dialogue):
            current_dialogue = dialogue_id
            history = []
            candidates = games_sets[current_dialogue]['items']

        # Reading each <question, answer>
        question = row['question']
        answer = row['answer']

        # if("correct" not in answer): # Excluding last dialogue question / answer
        history.append(
            f"-{question} {answer}"
        )

        # remaining_candidates = list_remaining_candidates(history, candidates)
        # print(remaining_candidates)
        # candidates = remaining_candidates

        score_list = {}
        remaining_candidates = defaultdict(list)
        for k in range(samples):
            remaining_candidates[k] = list_remaining_candidates(model, history, candidates)
            # print(remaining_candidates[k])

            if remaining_candidates[k]:
                p = 1 / len(remaining_candidates[k])

                for item in remaining_candidates[k]:
                    if item in score_list.keys():
                        score_list[item] = score_list[item] + p
                    else:
                        score_list[item] = p

        normalized_scores = {}
        scores_sum = 0
        for item in score_list.keys():
            scores_sum = scores_sum + score_list[item]

        for item in score_list.keys():
            normalized_scores[item] = round(score_list[item] / scores_sum, 4)

        # Computing probability
        # if candidates != []:
        # p = 1 / len(remaining_candidates)
        # else:
        # p = 0

        dump_row(
            dump_path,
            dialogue_id,
            row["intra_dialogue_id"],
            row["target"],
            question,
            answer,
            list(remaining_candidates.values()),
            score_list,
            normalized_scores
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


def openai_handler(model, prompt, temperature):
    return openai.chat.completions.create(
        temperature=temperature,
        model=model,
        messages=[{'role': "system", 'content': prompt}],
    ).choices[0].message.content


def ollama_handler(model, prompt, temperature):
    return ollama.chat(model=model, options={"temperature": temperature},
                       messages=[{'role': 'system', 'content': prompt}])['message']['content']


def generate_response(model_name, prompt, temperature=0.2):
    model_map = {
        'gpt3': {'model': 'gpt-3.5-turbo', 'handler': openai_handler},
        'gpt4': {'model': 'gpt-4o', 'handler': openai_handler},
        'llama': {'model': 'llama3', 'handler': ollama_handler}
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model name: {model_name}. Please use 'gpt3', 'gpt4', or 'llama'.")

    model_info = model_map[model_name]
    model = model_info['model']
    handler = model_info['handler']

    response = handler(model, prompt, temperature)

    return response


def open_step_by_step_csv(data_path):
    with open(data_path, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow([
            "dialogue_id",  # enumeration of games dialogues
            "intra_dialogue_id",  # question/answer index inside dialogue with id = dialogue_id
            "target",  # item assigned to user
            "question",  # question made by the guesser
            "answer",
            "candidates",
            "candidates_scores",
            "p_distribuition",
        ])


def open_game_sets(data_path):
    with open(f"{data_path}/contrast_sets.json") as f:
        return json.load(f)


def open_dialogues(data_path):
    rows = []
    with open(data_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            rows.append(row)

    return rows


def list_remaining_candidates(model, history, candidates):
    history_str = "\n".join(history)
    candidates_str = ", ".join(candidates)
    prompt = build_prompt(candidates_str, history_str)

    while True:
        # Asking model to list out the remaining items
        response = generate_response(model, prompt)
        pattern = r"CANDIDATES: ([^;\n]+)"
        matches = re.findall(pattern, response)
        candidates = set()
        for match in matches:
            candidates_list = match.split(", ")
            cleaned_candidates = {remove_punctuation(candidate).strip() for candidate in candidates_list}
            candidates.update(cleaned_candidates)

        if all(len(candidate.split()) == 1 for candidate in candidates):
            break
        else:
            print("Respo:", candidates)
        
    # explanation = response.split("EXPLANATION: ")[1]
    return candidates


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def dump_row(data_path, dialogue_id, intra_dialogue_id, target, question, answer, candidates, score, score_prob):
    with open(data_path, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow([
            dialogue_id,
            intra_dialogue_id,
            target,
            question,
            answer,
            candidates,
            score,
            score_prob
        ])


if __name__ == "__main__":
    main()
