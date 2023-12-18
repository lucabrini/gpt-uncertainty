
import asyncio
import json
from os import environ
from dotenv import load_dotenv
import openai

from uncertainty.utils import OpenAIModelEnum
from uncertainty.custom_model import LLModelWrapper

from questions.generate_dialogues import generate_dialogues_openai, get_lists_of_candidates

if __name__ == "__main__":
  
  load_dotenv()
  openai_api_key = environ.get("OPENAI_API_KEY")
  
  openai.api_key = openai_api_key   
  
  model = LLModelWrapper(
    api_key=openai_api_key, 
    model=OpenAIModelEnum.GPT_3_5_TURBO,
    debug=True
  )
  
  answer, confidence = asyncio.run(
    model.ask({
      "role" : "user",
      "content": "What's the capital of Italy?"
    })
  )
  
  with open(f"../data/game_sets/contrast_sets.json") as f:
    contrast_sets = json.load(f)
    
  num_candidates = 8

  target_list_candidates = get_lists_of_candidates(contrast_sets)

  generate_dialogues_openai(model, target_list_candidates, num_candidates)
  
  
  
  
  print("Answer: ", answer)
  print("Confidence: ", confidence)