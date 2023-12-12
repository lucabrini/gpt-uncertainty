from dotenv import load_dotenv
from os import environ

from custom_model import CustomModel
from openai_model_enum import OpenAIModelEnum

if __name__ == "__main__":
  
  load_dotenv()
  openai_api_key = environ.get("OPENAI_API_KEY")
  
  model = CustomModel(
    api_key=openai_api_key, 
    model=OpenAIModelEnum.GPT_3_5_TURBO
  )
  
  model.ask("What is 2+2?")