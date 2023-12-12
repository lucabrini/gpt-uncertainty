from openai import OpenAI
from openai_model_enum import OpenAIModelEnum
from build_prompts import *
import re
from nli import compute_mean_similarity

class CustomModel:
  
  
  def __init__(self, api_key: str, model: OpenAIModelEnum, alpha=0.8, k=5) -> None:
      openai = OpenAI(api_key=api_key)
      
      self.model = model.value
      self.llm = openai.chat.completions
      
      self.alpha = alpha
      self.k = k
      
      
  def ask(self, question):
    
    original_answer = self.llm.create(
      model=self.model,
      temperature=0,
      messages=[
        {
            "role": "user",
            "content": question
        },
      ],
    )
    
    original_answer = original_answer.choices[0].message.content
    y = self.extract_cot_answer(original_answer)
    
    # Producing Diverse Output
    multiple_outputs = self.observe_consistency(question, self.k)
    
    # Measuring Similarity between Sampled and Original Answer
    observed_consistency = 0
    for raw_output in multiple_outputs:
      
      # Extract answer from templated response
      yi = self.extract_cot_answer(raw_output)
      si = compute_mean_similarity(y, yi)
      
      # Indicator function
      ri = 1 if y == yi else 0
      
      # Similarity Score
      oi = self.alpha * si + (1-self.alpha)*ri
      observed_consistency += oi
      
    return y, observed_consistency/self.k
      
      
  def observe_consistency(self, question: str, k=5) -> list[str]:
    
    answers = []
    for _ in range(0,k,1):
      
      prompt = build_observed_consistency_prompt(question)
      print(prompt)
      response = self.llm.create(
        model=self.model,
        temperature=1,
        messages=[
          {
              "role": "user",
              "content": prompt
          },
        ],
      )
      
      answers.append(response.choices[0].message.content)
      
    return answers
  
  
  def self_reflection(self, question: str, proposed_answer: str):
    
    response = self.llm.create(
        model=self.model,
        temperature=1,
        messages=[
          {
              "role": "user",
              "content": build_self_reflection_certainty_prompt(question, proposed_answer)
          },
        ],
      )
      
    return response.choices[0].message.content
  
  
  def extract_cot_answer(self, response: str):
    regexp = r"answer:\s*(.+)"
    
    match = re.search(regexp, response)

    if match:
        return match.group(1)
    else:
        return ""