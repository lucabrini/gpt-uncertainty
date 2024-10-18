import re
from string import Template
import openai
import ollama

    
def compute_item_probability(history, candidates, openai_api_key=None, samples=5, model_name='llama3'):
  
  if openai_api_key:
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
      while True:
        response = generate_response(model_name, candidate, history, temperature=0)

        explanation, answer = extract_explanation_and_answer(response)

        if answer is None or len(answer.split()) == 1:
           break
        else:
           print(response)
           print("\n", answer)

      explanations[candidate].append(explanation)

      if answer is not None and "yes" in answer.lower():
        yes_counter += 1
      
      scores[candidate] = round(yes_counter / samples, 4)
      
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
  regex = r"EXPLANATION:\s*(.*?)\s*ANSWER:\s*.*\b([Yy][Ee][Ss]|[Nn][Oo])\b.*$"
  match = re.search(regex, response, re.DOTALL)
    
  if match:
      explanation = match.group(1).strip()
      answer = match.group(2).strip().lower()  # Convertiamo la risposta in minuscolo per uniformità
      return explanation, answer
  else:
      return None, None
  
def generate_response(model_name, candidate, history, temperature=0.2):
    model_map = {
        'gpt3': {'model': 'gpt-3.5-turbo', 'handler': openai_handler},
        'gpt4': {'model': 'gpt4o', 'handler': openai_handler},
        'llama': {'model': 'llama3', 'handler': ollama_handler}
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model name: {model_name}. Please use 'gpt3', 'gpt4', or 'llama'.")

    model_info = model_map[model_name]
    model = model_info['model']
    handler = model_info['handler']

    response = handler(model, candidate, history, temperature)

    return response


def openai_handler(model, candidate, history, temperature):
    return openai.chat.completions.create(
        temperature=temperature,
        model=model,
        messages=[{"role": "system", "content": build_prompt(candidate, history)}]
    ).choices[0].message.content


def ollama_handler(model, candidate, history, temperature):
    return ollama.chat(
        model=model, 
        options={"temperature": temperature}, 
        messages=[{"role": "system", "content": build_prompt(candidate, history)}]
    )['message']['content']
