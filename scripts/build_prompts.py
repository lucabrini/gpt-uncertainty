from string import Template

def build_observed_consistency_prompt(question: str):
  template = Template(
    """
      Strictly use the following template to provide answer, even casing: 
        explanation: [insert step-by-step analysis], 
        answer: [provide your answer]
          
      Question: [$question]
    """
  )
  
  return template.substitute(question=question)

def build_self_reflection_certainty_prompt(question: str, proposed_answer: str):
  template = Template(
      """
        Answer to the following bullets point separately.
        
        1. Question: [$question], Proposed Answer: [$proposed_answer]. Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template, even casing: 
          explanation: [insert analysis], 
          answer: [choose and write only one letter from among choices A through C]
        
        2. Question: [$question], Proposed Answer: [$proposed_answer]. Are you really sure the proposed answer is correct? Choose again: (A) Correct (B) Incorrect (C) I am not sure. The output should strictly use the following template, even casing: 
          explanation: [insert analysis], 
          answer: [choose and write only one letter from among choices A through C]
      """
    )


  return template.substitute(question=question, proposed_answer=proposed_answer)

    



  

