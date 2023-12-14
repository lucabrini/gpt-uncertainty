from openai import AsyncOpenAI
from utils import OpenAIModelEnum, self_reflection_answers_mapping
from build_prompts import *
import re
from nli import compute_mean_similarity

class CustomModel:
  
  
  def __init__(self, api_key: str, model: OpenAIModelEnum, alpha=0.8, beta=0.7, k=5) -> None:
      openai = AsyncOpenAI(api_key=api_key)
      
      self.model = model.value
      self.llm = openai.chat.completions
      
      self.alpha = alpha
      self.beta = beta
      self.k = k
      
    
      
  async def ask(self, question):
    
    print("# Original Answer\n")
    original_answer = await self.llm.create(
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
    confidence = await self.run_bsdetector(question, original_answer)
      
    return original_answer, confidence
      
      
      
  async def run_bsdetector(self, question, original_answer):
    print("# Running BSDetector\n")
    observed_consistency = await self.observe_consistency(question, original_answer)
    self_reported_certainty = await self.self_reflection(question, original_answer)
    
    confidence = self.beta * observed_consistency + (1 - self.beta) * self_reported_certainty
    return confidence
      
      
      
  async def observe_consistency(self, question: str, original_answer) -> list[str]:
    print("# Observing consistency:")
    sampled_multiple_outputs = await self.sample_multiple_outputs(question)
    
    observed_consistency = 0
    for raw_output in sampled_multiple_outputs:
      
      # Extract answer from templated response
      print(raw_output)
      print("-----")
      yi = self.extract_llm_answer(raw_output)[0]
      si = compute_mean_similarity(original_answer, yi)
      
      # Indicator function
      ri = 1 if original_answer == yi else 0
      
      # Similarity Score
      oi = self.alpha * si + (1-self.alpha)*ri
      observed_consistency += oi
    
    observed_consistency = observed_consistency/self.k 
    print("# Observed consistency: " + str(observed_consistency) + "\n")
    return observed_consistency
  
  
  
  async def sample_multiple_outputs(self, question: str):
    print("# Sampling multiple outputs:\n")
    answers = []
    for i in range(0,self.k,1):
      print(i, ",")
      prompt = build_observed_consistency_prompt(question)
      response = await self.llm.create(
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
    print(",")
    return answers
  
  
  
  async def self_reflection(self, question: str, proposed_answer: str) -> float:
    print("# Self Reflection:")
    prompt = build_self_reflection_certainty_prompt(question, proposed_answer)
    
    response = await self.llm.create(
      model=self.model,
      temperature=1,
      messages=[
        {
            "role": "user",
            "content": prompt
        },
      ],
    )
      
    answers = self.extract_llm_answer(response.choices[0].message.content)[0]
    
    self_reflection = 0  
    for a in answers:
      self_reflection += self_reflection_answers_mapping[a]

    self_reflection = self_reflection/len(answers)
    print(self_reflection)
    print("\n")
    return self_reflection/len(answers)
  
  
  
  def extract_llm_answer(self, response: str) -> list[str]:
    regexp = r"answer:\s*(.+)"
    
    match = re.findall(regexp, response)
    return match
      