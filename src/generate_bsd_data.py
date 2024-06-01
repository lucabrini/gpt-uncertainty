import argparse
import json
import os
import re
from dotenv import load_dotenv

from questions_game.scripts.generate_dialogues import generate_dialogues_openai, get_lists_of_candidates
from bsd_detector.utils import OpenAIModelEnum
from bsd_detector.custom_model import LLModelWrapper

if __name__ == "__main__":
    
    load_dotenv()
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument("--game_set", type=str, default="8_mcrae",
        choices=["8_mcrae", "16_mcrae", "8_gpt", "8_mcrae_stepwise", "8_wordnet"])
    args = parser.parse_args()
    
    model = LLModelWrapper(
      api_key=openai_api_key, 
      model=OpenAIModelEnum.GPT_4_O,
      debug=True
    )

    game_set = "8_mcrae" if args.game_set == "8_mcrae_stepwise" else args.game_set

    with open(f"./data/game_sets/{game_set}/contrast_sets.json") as f:
        contrast_sets = json.load(f)
    
    num_candidates = int(re.sub(r"_.*", "", args.game_set))

    target_list_candidates = get_lists_of_candidates(contrast_sets)
    #print(target_list_candidates)
    
    generate_dialogues_openai(model, target_list_candidates, game_set, num_candidates, bsd=False, outfile_name="dialogues-gpt-4o")