import re
from string import Template
import openai

    
def compute_item_probability(history, candidates, openai_api_key, samples=5):
  
  openai_client = openai.OpenAI(
    api_key=openai_api_key
  )
  
  items_p_distribuition = []
  

  history = "\n".join(history)
  
  scores = {}
  explanations = {}
  
  for candidate in candidates:
    
    yes_counter = 0
    
    explanations[candidate] = []
    
    for _ in range(0, samples, 1):
      
      response = openai_client.chat.completions.create(
        temperature=0,
        model='gpt-3.5-turbo',
        messages=[
          {"role": "system", "content": build_prompt(candidate, history)},
        ],
      ).choices[0].message.content

      explanation, answer = extract_explanation_and_answer(response)
      explanations[candidate].append(explanation)
      print(response)
      if("yes" in answer.lower()):
        yes_counter += 1
      
      scores[candidate] = round(yes_counter / samples, 4)
    
  normalized_scores = {}
  scores_sum = 0
  for candidate in candidates:
    scores_sum = scores_sum + scores[candidate]
    
  for candidate in candidates:
    normalized_scores[candidate] = round(scores[candidate] / scores_sum, 4)
  
  items_p_distribuition = {
    "normalized_scores" : normalized_scores, 
    "scores" : scores,
    "explanations" : explanations
  }
    
  return items_p_distribuition


# Utilities

def build_prompt(candidate, dialogue):  
  template = Template(
    "Given this dialogue: \n"
    "$dialogue"
    "\n"
    "Is the dialogue true for this item? \n"
    "Item: $candidate \n"
    "\n"
    "The output must strictly use the following template: \n"
    "EXPLANATION: [insert your analysis]"
    "ANSWER: [is the dialogue true, yes or no] \n"
  )  
  
  return template.substitute(candidate=candidate, dialogue=dialogue)

  
def extract_explanation_and_answer(response):
  regex = r"EXPLANATION:\s*(.*?)\s*ANSWER:\s*(.*)$"
  match = re.search(regex, response, re.IGNORECASE | re.DOTALL)
  
  if match:
    explanation = match.group(1).strip()
    answer = match.group(2).strip()
    return explanation, answer
  else:
      return None, None
    