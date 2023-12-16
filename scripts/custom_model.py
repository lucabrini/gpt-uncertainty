from openai import AsyncOpenAI
from utils import OpenAIModelEnum, self_reflection_answers_mapping
from build_prompts import *
import re
from nli import compute_mean_similarity

class CustomModel:
  
  
  def __init__(self, api_key: str, model: OpenAIModelEnum, alpha=0.8, beta=0.7, k=5, debug=False) -> None:
      openai = AsyncOpenAI(api_key=api_key)
      
      self.model = model.value
      self.llm = openai.chat.completions
      
      self.alpha = alpha
      self.beta = beta
      self.k = k
      
      self.debug = debug
      
    
      
  async def ask(self, question):
    
    self.debug_log("# Asking for the Original Answer\n")
    prompt = build_original_question_prompt(question)
    original_answer = await self.llm.create(
      model=self.model,
      temperature=0,
      messages=[
        {
            "role": "system",
            "content": prompt
        },
      ],
    )
    
    original_answer = self.extract_llm_answer(original_answer.choices[0].message.content)[0]
    confidence = await self.run_bsdetector(question, original_answer)
      
    return original_answer, confidence
      
      
      
  async def run_bsdetector(self, question, original_answer):
    self.debug_log("# Running BSDetector")
    observed_consistency = await self.observe_consistency(question, original_answer)
    self_reported_certainty = await self.self_reflection(question, original_answer)
    
    confidence = self.beta * observed_consistency + (1 - self.beta) * self_reported_certainty
    return confidence
      
      
      
  async def observe_consistency(self, question: str, original_answer) -> list[str]:
    self.debug_log("\tObserving consistency:")
    sampled_multiple_outputs = await self.sample_multiple_outputs(question)
    
    observed_consistency = 0
    self.debug_log("\t\t Similarities")
    for i in range(0, len(sampled_multiple_outputs), 1):
      
      raw_output = sampled_multiple_outputs[i]
      # Extract answer from templated response
      yi = self.extract_llm_answer(raw_output)[0]
      si = compute_mean_similarity(question, original_answer, yi)
      self.debug_log("\t\t\t# " + str(i) + "\n\t\t\t\tAnswer: " + str(yi) + "\n\t\t\t\tSimilarity: " + str(si)  )
      
      # Indicator function
      ri = 1 if original_answer == yi else 0
      
      # Similarity Score
      oi = self.alpha * si + (1-self.alpha)*ri
      observed_consistency += oi
    
    observed_consistency = observed_consistency/self.k 
    self.debug_log("\t\t Observed consistency: " + str(observed_consistency) + "\n")
    return observed_consistency
  
  
  
  async def sample_multiple_outputs(self, question: str):
    self.debug_log("\t\t Sampling multiple outputs: ", end="")
    
    answers = []
    for i in range(0,self.k,1):
      self.debug_log(str(i) + "...", end="")
      prompt = build_observed_consistency_prompt(question)
      response = await self.llm.create(
        model=self.model,
        temperature=1,
        messages=[
          {
              "role": "system",
              "content": prompt
          },
        ],
      )
      answers.append(response.choices[0].message.content)
    self.debug_log("\n")
    return answers
  
  
  
  async def self_reflection(self, question: str, proposed_answer: str) -> float:
    self.debug_log("# Self Reflection:")
    prompt = build_self_reflection_certainty_prompt(question, proposed_answer)
    
    self.debug_log("\t Asking the LLM about the proposed answer:")
    response = await self.llm.create(
      model=self.model,
      temperature=1,
      messages=[
        {
            "role": "system",
            "content": prompt
        },
      ],
    )
    
    raw_response = response.choices[0].message.content
    self.debug_log("\t\t", raw_response)
    
    answers = self.extract_llm_answer(raw_response)
    
    self_reflection = 0  
    for a in answers:
      self_reflection += self_reflection_answers_mapping[a]

    self_reflection = self_reflection/len(answers)
    self.debug_log("\tSelf Reflection: " + str(self_reflection))
    return self_reflection/len(answers)
  
  
  
  def extract_llm_answer(self, response: str) -> list[str]:
    print(response)
    regexp = r"answer:\s*(.+)"
    
    match = re.findall(regexp, response)
    return match
      
      
  def debug_log(self, *args, **kwargs):
    if(self.debug):
      print(*args, **kwargs)