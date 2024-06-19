import csv
import json
import math
import numpy as np
import ast
from collections import Counter

data_path = "./src/data/generation/8_mcrae/dialogues(gpt3)_app2_onllama3_k5.csv"
filename = "entropy(gpt3)_app2_onllama3_k5_apocalypse_cleaned"

to_clean = True


def main():
    rf = open(data_path, 'r', newline='')
    reader = csv.DictReader(rf, delimiter=",")
    zeros_list = np.zeros(8)
    
    with open(f"./src/data/generation/8_mcrae/{filename}.csv", "w", newline='') as df:
        csv.writer(df).writerow([
            "dialogue_id",
            "intra_dialogue_id",
            "entropy"
        ])

    previous_scores = []
    for row in reader:
        raw_scores = row["candidates_scores"].replace('\'', '"')
        raw_distribuition = row["p_distribuition"].replace('\'', '"')
        row_k_candidates = row["candidates"].replace('\'', '"').replace('set()', '{None}')
        
        if len(raw_scores) != 0:
            json_scores = json.loads(raw_scores)
            json_distribuition = json.loads(raw_distribuition)
            # Rimuovi i caratteri esterni della lista e dividi la stringa in componenti
            elements = row_k_candidates[1:-1].split('}, {')

            # Aggiusta le parentesi mancanti e converte ogni componente in un insieme
            list_of_sets = [set(ast.literal_eval('{' + element.strip('{}') + '}')) if element.strip('{}') != 'None' else None for element in elements]

            # Conta le occorrenze di ogni elemento
            counter = Counter()
            for s in list_of_sets:
                counter.update(s)

            # Converti il Counter in un dizionario
            k_candidates = dict(counter)

            scores = list(json_scores.values())
            distr = list(json_distribuition.values())
            list_candidates = list(json_distribuition.keys())

            if (row["intra_dialogue_id"] == "0"):
                previous_scores = scores
                previous_distr = distr
                previous_list_candidates = list_candidates

            if (to_clean):
                distr, list_candidates = clean_scores(list_candidates, previous_list_candidates, k_candidates)
                previous_list_candidates = list_candidates

            entropy = 0
            for c in distr:
                if (c != 0):
                    entropy = entropy + c * math.log(c, 2)
            entropy = round(-1 * entropy, 4)
            print(entropy)
        else:
            entropy = ''
        with open(f"./src/data/generation/8_mcrae/{filename}.csv", "a", newline='') as df:
            csv.writer(df).writerow([
                row["dialogue_id"],
                row["intra_dialogue_id"],
                entropy
            ])
    rf.close()


# This functions takes as argument the previous (normalized) scores and the current ones readen from the csv
# and excludes the current candidates (from the current scores) which have been excluded in the previous steps
def clean_scores(list_candidates, previous_list_candidates, result_dict):
    # Verifica quali candidati non sono presenti nella lista precedente
    new_candidates = [item for item in list_candidates if item not in previous_list_candidates]

    # Filtra il dizionario dei risultati per rimuovere gli elementi non presenti
    cleaned_distribution = {item: result_dict[item] for item in result_dict if item not in new_candidates}

    # Calcola la somma dei valori nella distribuzione pulita
    total_count = sum(cleaned_distribution.values())

    # Normalizza e approssima la distribuzione alla quarta cifra decimale
    normalized_distribution = {item: round(count / total_count, 4) for item, count in cleaned_distribution.items()}

    # Nuova lista di candidati che esclude i nuovi candidati
    new_list_candidates = [item for item in list_candidates if item not in new_candidates]

    return normalized_distribution.values(), new_list_candidates


main()
