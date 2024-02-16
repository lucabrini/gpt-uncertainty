from string import Template
import openai

    
def compute_item_probability(history, candidates, openai_api_key, samples=5):
  
  openai_client = openai.OpenAI(
    api_key=openai_api_key
  )
  
  items_p_distribuition = []
  

  history = "\n".join(history)
  
  scores = {}
  for candidate in candidates:
  
    yes_counter = 0
    for _ in range(0, samples, 1):
      
      response = openai_client.chat.completions.create(
        temperature=0,
        model='gpt-3.5-turbo',
        messages=[
          {"role": "system", "content": build_prompt(candidate, history)},
        ],
      ).choices[0].message.content
    
      print(candidate, response)
    
      # TODO: regex to extract the ANSWER key and its value
      if("yes" in response.lower()):
        yes_counter += 1
      
      scores[candidate] = yes_counter / samples
    
  normalized_scores = {}
  scores_sum = 0
  for candidate in candidates:
    scores_sum = scores_sum + scores[candidate]
    
  for candidate in candidates:
    normalized_scores[candidate] = scores[candidate] / scores_sum
  
  items_p_distribuition = {"normalized_scores" : normalized_scores, "scores" : scores}
    
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
    "ANSWER: [is the dialogue true] \n"
  )  
  
  return template.substitute(candidate=candidate, dialogue=dialogue)

  