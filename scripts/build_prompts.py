from string import Template

def build_original_question_prompt(question: str):
  return build_observed_consistency_prompt(question)

def build_observed_consistency_prompt(question: str):
  template = Template(
    """
      Strictly use the following template to provide answer, answer key must be lowercase: 
        explanation: [insert step-by-step analysis], 
        answer: [provide your answer]
          
      Question: [$question]
    """
  )
  
  return template.substitute(question=question)

def build_self_reflection_certainty_prompt(question: str, proposed_answer: str):
  template = Template(
      """
        Answer to the following bullets point.
        
        1. Question: [$question], Proposed Answer: [$proposed_answer]. 
          Is the proposed answer: 
          - (A) Correct (if the proposed answer is similar to yours)
          - (B) Incorrect 
          - (C) I am not sure. 
          
          The output should strictly use the following template, answer key must be lowercase: 
            explanation: [insert analysis], 
            answer: [choose one letter from among choices A through C, without parenthesis or anything else]
        
        2. Question: [$question], Proposed Answer: [$proposed_answer]. 
          Are you really sure the proposed answer is correct? Choose again: 
          - (A) Correct (if the proposed answer is similar to yours)
          - (B) Incorrect 
          - (C) I am not sure. 
          
          The output should strictly use the following template, answer key must be lowercase: 
            explanation: [insert analysis], 
            answer: [choose one letter from among choices A through C, without parenthesis or anything else. The letter must be uppercase]
      """
    )


  return template.substitute(question=question, proposed_answer=proposed_answer)

    



  

