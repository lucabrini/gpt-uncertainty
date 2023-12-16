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
    debug=False
  )
  
  answer, confidence = asyncio.run(
    model.ask("What's the capital of Italy?")
  )
  
  print("Answer: ", answer)
  print("Confidence: ", confidence)