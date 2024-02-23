from openai import OpenAI
import re
from bsd_detector.build_prompts import build_observed_consistency_prompt, build_self_reflection_certainty_prompt
from bsd_detector.nli import compute_mean_similarity
from bsd_detector.utils import OpenAIModelEnum

from utils import self_reflection_answers_mapping

class LLModelWrapper:
  
  
  def __init__(self, api_key: str, model: OpenAIModelEnum, alpha=0.8, beta=0.7, k=5, debug=False) -> None:
      openai = OpenAI(api_key=api_key)
      
      self.model = model.value
      self.llm = openai.chat.completions
      
      self.alpha = alpha
      self.beta = beta
      self.k = k
      
      self.debug = debug
      
    
      
  def ask(self, question, message_history=[], role="system", temperature=1):
    
    self.debug_log("# Asking for the Original Answer\n")
    original_answer = self.llm.create(
      model=self.model,
      temperature=temperature,
      messages=[
        *message_history,
        {
          "role" : role,
          "content" : question
        }
      ]
    )
    
    original_answer = original_answer.choices[0].message.content
    uncertainity_metrics = self.run_bsdetector(question, original_answer, message_history)
    
    return original_answer, uncertainity_metrics
      
      
      
  def run_bsdetector(self, question, original_answer, message_history):
    self.debug_log("# Running BSDetector")
    
    observed_consistency = self.observe_consistency(question, original_answer, message_history)
    self_reported_certainty = self.self_reflection(question, original_answer, message_history)
    
    confidence = self.beta * observed_consistency + (1 - self.beta) * self_reported_certainty
    
    uncertainity_metrics = {
      "confidence" : confidence,
      "observed_consistency" : observed_consistency,
      "self_reported_certainty" : self_reported_certainty,
    }
    
    return uncertainity_metrics
      
      
      
  def observe_consistency(self, question: str, original_answer, message_history) -> list[str]:
    self.debug_log("\t Observing consistency:")
    sampled_multiple_answers = self.sample_multiple_outputs(question, message_history)
    
    observed_consistency = 0
    self.debug_log("\t\t Similarities")
    for i, yi in enumerate(sampled_multiple_answers):
    
      si = compute_mean_similarity(question, original_answer, yi)
      self.debug_log("\t\t\t  # " + str(i) + "\n\t\t\t\tAnswer: " + str(yi) + "\n\t\t\t\tSimilarity: " + str(si)  )
      
      # Indicator function
      ri = 1 if original_answer == yi else 0
      
      # Similarity Score
      oi = self.alpha * si + (1-self.alpha)*ri
      observed_consistency += oi
    
    observed_consistency = observed_consistency/self.k 
    self.debug_log("\t\t Observed consistency: " + str(observed_consistency) + "\n")
    return observed_consistency
  
  
  
  def sample_multiple_outputs(self, question: str, message_history):
    self.debug_log("\t\t Sampling multiple outputs: ", end="")
    
    answers = []
    for i in range(0,self.k,1):
      self.debug_log(str(i) + "...", end="")
      
      prompt = build_observed_consistency_prompt(question)
      
      success = False
      while not success:
        response = self.llm.create(
          model=self.model,
          temperature=1,
          messages=[
            *message_history,
            {
                "role": "system",
                "content": prompt
            },
          ],
        )
      
        try:
          yi = self.extract_llm_answer(response.choices[0].message.content)[0]
          success = True
          answers.append(yi)
        except IndexError:
          pass
    
    self.debug_log("\n")
    return answers
  
  
  
  def self_reflection(self, question: str, proposed_answer: str, message_history) -> float:
    self.debug_log("# Self Reflection:")
    
    prompt = build_self_reflection_certainty_prompt(question, proposed_answer)
    
    self.debug_log("\t Asking the LLM about the proposed answer:")
    
    print( [*message_history,
          {
              "role": "system",
              "content": prompt
          },])
    
    while True:
      response = self.llm.create(
        model=self.model,
        temperature=1,
        messages=[
          *message_history,
          {
              "role": "system",
              "content": prompt
          },
        ],
      )
      
      raw_response = response.choices[0].message.content
      self.debug_log("\t\t", raw_response)
      

      answers = self.extract_llm_answer(raw_response)
      if(len(answers) != 0):

        self_reflection = 0  
        try:
          for a in answers:
            self_reflection += self_reflection_answers_mapping[a]
          break
        except KeyError:
          pass
        

    self_reflection = self_reflection/len(answers)
    self.debug_log("\t Self Reflection: " + str(self_reflection))
    return self_reflection
  
  
  
  def extract_llm_answer(self, response: str, ) -> list[str]:
    regexp =  re.compile(r"response:\s*(.+)", re.IGNORECASE) 
    match = re.findall(regexp, response)
    
    return match
      
      
  def debug_log(self, *args, **kwargs):
    if(self.debug):
      print(*args, **kwargs)