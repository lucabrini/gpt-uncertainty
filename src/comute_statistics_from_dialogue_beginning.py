import csv
import numpy

from utils import group_entropies_by_dialogue_id

data_path = "./data/generation/8_mcrae/sbs_entropy_k_ten_cleaned.csv"
filename = "entropy_k_ten_statistics_from_dialogue_beginning_cleaned"


def main():
    rf = open(data_path, 'r', newline='')
    reader = csv.DictReader(rf, delimiter=",")

    dialogues_entropies, max_dialogue_length = group_entropies_by_dialogue_id(reader)

    entropies_by_distances = []

    for _ in range(0, max_dialogue_length, 1):
        entropies_by_distances.append([])

    for dialogue_entropy in dialogues_entropies.values():
        for step_index, step_entropy in enumerate(dialogue_entropy):
            entropies_by_distances[step_index].append(step_entropy)

    with open(f"./data/generation/8_mcrae/{filename}.csv", "w", newline='') as df:
        csv.writer(df).writerow([
            "distance",
            "mean",
            "std"
        ])

    for (distance, entropies_by_distance) in enumerate(entropies_by_distances):
        std = numpy.std(entropies_by_distance)
        mean = numpy.mean(entropies_by_distance)
        print(distance, std, mean)

        with open(f"./data/generation/8_mcrae/{filename}.csv", "a", newline='') as df:
            csv.writer(df).writerow([
                distance, std, mean
            ])


main()