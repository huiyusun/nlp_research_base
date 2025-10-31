from transformers import BertTokenizerFast, BertForSequenceClassification
import torch, json

ckpt = "outputs/bert-tacred"
tok = BertTokenizerFast.from_pretrained(ckpt)
model = BertForSequenceClassification.from_pretrained(ckpt)
id2label = model.config.id2label

ex = {
    "tokens": ["Barack", "Obama", "was", "born", "in", "Hawaii", "."],
    "h": {"pos": [0, 1], "type": "PERSON"},
    "t": {"pos": [5, 5], "type": "LOCATION"}
}
from train_tacred_bert import mark_entities

text = mark_entities(ex["tokens"], ex["h"]["pos"], ex["t"]["pos"], ex["h"]["type"], ex["t"]["type"])
enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**enc).logits
pred = logits.argmax(-1).item()
print(id2label[pred])
