import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base").to("mps")

def compute_mean_similarity(y : str, yi : str) -> float:
  
  p = compute_similarity(y, yi)
  p1 = compute_similarity(yi, y)
  
  #si = (1-p + 1-p1) / 2
  return 0
    

def compute_similarity(x: str, y: str):

  sequence = (x, y)
  
  encoded_input = tokenizer.encode(sequence, padding=True)
  
  prediction = model(torch.tensor(torch.tensor([encoded_input]), device='mps'))['logits']

  probabilities = torch.argmax(prediction, dim=1)
  
  #p_contradiction = probabilities[0, tokenizer.convert_tokens_to_ids("contradiction")]
  print("\n\n\n\n")
  print(probabilities)
  
  
compute_mean_similarity("The capital of France is Paris", "The France's Capital is Paris")