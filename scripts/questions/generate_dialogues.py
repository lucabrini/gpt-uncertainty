from dotenv import load_dotenv
import os
import time
import random
import re 
import argparse
from functools import wraps
import json
import openai

from uncertainty.custom_model import LLModelWrapper

def retry_on_rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except openai.error.RateLimitError:
                print("Rate limit reached. Waiting for 10 seconds...")
                time.sleep(10)
    return wrapper

def get_lists_of_candidates(contrast_sets):
  list_and_target = {}
  count_ = 0
  for contrast_set in contrast_sets.values():
    list_and_target[count_] = {'candidates':contrast_set['items'], 'target':contrast_set['target']}
    count_ += 1
  return list_and_target

def openai_call(conversation, oracle=False):
    if oracle:
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=conversation,
            temperature=0.1,
        )
    else:
        
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=conversation,
        )
    return {'role': response.choices[0].message.role, 'content': response.choices[0].message.content}

def get_prompts(candidates, target, stepwise=False):
    if stepwise:
        questioner = ([{'role': "system", 'content': "You are playing an interactive game with the user, who is assigned "\
                                                        "an item from a list of candidates. Ask as few questions as possible to identify the item, "\
                                                        "making only one question at each turn.\n"\
                                                        "\nThe user can only respond with 'yes' or 'no'.\n"\
                                                        "Format your output in the following way:\n"\
                                                        "CANDIDATES: item, item, item, item ...\n"\
                                                        "QUESTION: text of the question"},
                            {'role': "user", 'content': f"This is the list of candidates: {candidates}."}])
    else:
        questioner = ([{'role': "system", 'content': "You are playing an interactive game with the user, who is assigned "\
                                                        "an item from a list of candidates. Ask as few questions as possible to identify the item, "\
                                                        "making only one question at each turn.\n"\
                                                        "\nThe user can only respond with 'yes' or 'no'."},
                            {'role': "user", 'content': f"This is the list of candidates: {candidates}."}])
    
    oracle = ([{'role': "system", 'content': "You are playing an interactive game with the user, in which you are assigned one item from a list "\
                                                "of candidates."\
                                                "\nThe user will have to guess which one it is by asking yes/no questions, and "\
                                                "you have to respond to each question only with 'yes' or 'no'."\
                                                "\nIf the user correctly guesses your assigned item, respond with 'Yes! That's correct.'."\
                                                f"\nThe item assigned to you is {target}."}])
    return questioner, oracle


def generate_dialogues_openai(model: LLModelWrapper, target_list_candidates, num_candidates):

    if os.path.exists(f"../data/generation/dialogues.txt"):
        with open(f"../data/generation/dialogues.txt", "r") as f:
            dialogues_raw_txt = f.read()
            num_dialogues = len(dialogues_raw_txt.split("******************"))
            target_list_candidates = {key: target_list_candidates[key] for key in target_list_candidates.keys() if int(key) >= (num_dialogues - 1)}
            # print(target_list_candidates)
    else:
        if not os.path.exists(f"../data/generation/"):
            os.mkdir(f"../data/generation/")
        num_dialogues = 0

    stepwise = False

    for index, value in target_list_candidates.items():

        successful = False
        while not successful:

            dialogue = []

            target = value['target']

            print("******************")
            dialogue.append("******************")
            print(f"target = {target}")
            dialogue.append(f"target = {target}")

            # Initial prompts. Game rules
            questioner, oracle = get_prompts(", ".join(value['candidates']), target, stepwise=stepwise)
            
            print('answerer: {}\t'.format(questioner[-1]['content'].strip()))
            dialogue.append('answerer: {}'.format(questioner[-1]['content'].strip()))

            oracle_output = {"content" : ""}
            for interaction in range(20):
                questioner_output = openai_call(questioner)
                questioner.append({'role': 'assistant', 'content': re.sub(r"\n\n*", " ", questioner_output['content'])})
                try:
                    processed_questioner_output = questioner_output['content'].split("QUESTION:")[1].strip()
                except IndexError:
                    processed_questioner_output = questioner_output['content']
                
                oracle.append({'role': 'user', 'content': processed_questioner_output})
                print('questioner: {}\t'.format(questioner[-1]['content'].strip()))
                dialogue.append('questioner: {}'.format(questioner[-1]['content'].strip()))

                oracle_output = openai_call(oracle, oracle=True)
                questioner.append({'role': 'user', 'content': re.sub("\n", " ", oracle_output['content'])})
                oracle.append({'role': 'assistant', 'content': oracle_output['content']})
                print('answerer: {}\t'.format(questioner[-1]['content'].strip()))
                dialogue.append('answerer: {}'.format(questioner[-1]['content'].strip()))

                if "correct" in oracle_output['content'].lower() and "yes" in oracle_output['content'].lower():
                    with open(f'../data/generation/dialogues.txt', 'a') as f:
                        for line in dialogue:
                            f.write(f"{line}\n")
                    successful = True
                    break