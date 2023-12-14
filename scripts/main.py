import asyncio
from dotenv import load_dotenv
from os import environ

from custom_model import CustomModel
from utils import OpenAIModelEnum

if __name__ == "__main__":
  
  load_dotenv()
  openai_api_key = environ.get("OPENAI_API_KEY")
  
  model = CustomModel(
    api_key=openai_api_key, 
    model=OpenAIModelEnum.GPT_3_5_TURBO,
  )
  
  answer, confidence = asyncio.run(
    model.ask("A tower is made out of 4 blue blocks, twice as many yellow blocks, and an unknown number of red blocks. If there are 32 blocks in the tower in total, how many red blocks are there?")
  )
  
  print("Answer: ", answer)
  print("Confidence: ", confidence)