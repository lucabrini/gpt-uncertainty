import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification

device = "mps"

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

def compute_mean_similarity(context: str, y : str, yi : str) -> float:
  
  y = context + y
  yi = context + yi
  p = compute_probabilities(y, yi)
  p1 = compute_probabilities(yi, y)
  
  si = (1-p + 1-p1) / 2
  return si
    

def compute_probabilities(x: str, y: str):

  sequence = x + " [SEP] " + y
  encoded_input = tokenizer.encode(sequence, padding=True)
  
  outputs = model((torch.tensor([encoded_input]).to(device).clone().detach()))
  # prediction = outputs['logits']
  
  # predicted_label = torch.argmax(prediction, dim=1)
  predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  
  
  # print("Contradiction:", predicted_probability[2], "\n")
  # print("semantically different" if 0 in predicted_label else "semantically equals", "\n")
  
  torch.mps.empty_cache()
  
  return 1-predicted_probability[2]