import asyncio
from dotenv import load_dotenv
from os import environ

from custom_model import LLModelWrapper
from utils import OpenAIModelEnum

if __name__ == "__main__":
  
  load_dotenv()
  openai_api_key = environ.get("OPENAI_API_KEY")
  
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
  
  print("Answer: ", answer)
  print("Confidence: ", confidence)