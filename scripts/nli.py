import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to("mps")

def compute_mean_similarity(y : str, yi : str) -> float:
  
  p = compute_similarity(y, yi)
  p1 = compute_similarity(yi, y)
  
  #si = (1-p + 1-p1) / 2
  return 0
    

def compute_similarity(x: str, y: str):

  sequence = x + " [SEP] " + y
  
  encoded_input = tokenizer.encode(sequence, padding=True)
  
  prediction = model(torch.tensor(torch.tensor([encoded_input]), device="mps"))['logits']

  predicted_label = torch.argmax(prediction, dim=1)
  
  #p_contradiction = probabilities[0, tokenizer.convert_tokens_to_ids("contradiction")]
  
  print(predicted_label)
  print("semantically different" if 0 in predicted_label else "semantically equals")
  
  
  
compute_mean_similarity("the capital of France is Paris", "The France's Capital is Paris")