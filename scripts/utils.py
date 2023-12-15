from enum import Enum

class OpenAIModelEnum(Enum):
  GPT_3_5_TURBO="gpt-3.5-turbo"
  GPT_4="gpt-4"
  
  
self_reflection_answers_mapping = {
  "A" : 1.0,
  "B" : 0.0,
  "C" : 0.5,
  "a" : 1.0,
  "b" : 0.0,
  "c" : 0.5
}
