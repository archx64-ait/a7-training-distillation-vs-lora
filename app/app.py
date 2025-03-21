from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel
import torch

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEACHER = 'bert-base-uncased'
NUM_LABELS = 3

tokenizer = BertTokenizer.from_pretrained(TEACHER)

# user can choose between models
models = {
    'odd_layers': BertForSequenceClassification.from_pretrained('./saved_models/odd_layers').to(device),
    'even_layers': BertForSequenceClassification.from_pretrained('./saved_models/even_layers').to(device),
    'lora': PeftModel.from_pretrained(
        BertForSequenceClassification.from_pretrained(TEACHER, num_labels=NUM_LABELS),
        './saved_models/lora'
    ).to(device)
}

# set models to eval
for model in models.values():
    model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = 'lora'
    if request.method == 'POST':
        text = request.form['text']
        selected_model = request.form['model']
        model = models[selected_model]
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
        labels = {0: 'Hate Speech', 1: 'Offensive', 2: 'Neutral'}  # adjust if needed
        prediction = labels.get(pred, 'Unknown')
    return render_template('index.html', prediction=prediction, selected_model=selected_model)

if __name__ == '__main__':
    app.run(debug=True)
