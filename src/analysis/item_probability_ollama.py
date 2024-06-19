import ollama
import re
from string import Template

def compute_item_probability_ollama(history, candidates, samples=5):
    items_p_distribuition = []
    history = "\n".join(history)
    scores = {}
    explanations = {}

    for candidate in candidates:
        #print("history:",history)
        #print("candidate:", candidate)
        yes_counter = 0
        explanations[candidate] = []

        for _ in range(0, samples, 1):
           response = ollama.chat(model='llama3', options={"temperature": 0}, messages=[
              { 
                 'role': 'system',
                 'content': build_prompt(candidate, history),
              },
            ])
           explanation, answer = extract_explanation_and_answer(response['message']['content'])
           print("answer:", answer)
           explanations[candidate].append(explanation)

           if("yes" in answer.lower()):
              yes_counter += 1   
           #print("counter: ", yes_counter)

        scores[candidate] = round(yes_counter / samples, 4)
        print("Score: ",scores[candidate])

    normalized_scores = {}
    scores_sum = 0
    for candidate in candidates:
       scores_sum = scores_sum + scores[candidate]
       
    for candidate in candidates:
      if scores_sum !=0:
         normalized_scores[candidate] = round(scores[candidate] / scores_sum, 4)
      else:
         normalized_scores[candidate] = 0

    items_p_distribuition = {
       "normalized_scores" : normalized_scores, 
       "scores" : scores,
       "explanations" : explanations
    }

    return items_p_distribuition

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
  