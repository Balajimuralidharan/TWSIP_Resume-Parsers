import re
import fitz  # PyMuPDF
import docx
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
import torch
from flask import Flask, request, jsonify

def convert_pdf_to_text(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page in document:
        text += page.get_text()
    return text

def convert_doc_to_text(doc_path):
    doc = docx.Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Example usage
pdf_text = convert_pdf_to_text('resume.pdf')
doc_text = convert_doc_to_text('resume.docx')
cleaned_text = preprocess_text(pdf_text)
print(cleaned_text)


# Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)

def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, is_split_into_words=True)
    label_ids = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids.append([])
        for word_idx in word_ids:
            if word_idx is None:
                label_ids[i].append(-100)
            elif word_idx != previous_word_idx:
                label_ids[i].append(label[word_idx])
            else:
                label_ids[i].append(-100)
            previous_word_idx = word_idx
    tokenized_inputs['labels'] = label_ids
    return tokenized_inputs

# Example data
texts = [["this", "is", "a", "resume", "text"]]
labels = [[0, 0, 0, 1, 0]]  # 1 for the entity, 0 otherwise

tokenized_inputs = tokenize_and_align_labels(texts, labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_inputs):
        self.tokenized_inputs = tokenized_inputs

    def __len__(self):
        return len(self.tokenized_inputs['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_inputs.items()}
        return item

train_dataset = ResumeDataset(tokenized_inputs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

app = Flask(__name__)

@app.route('/parse_resume', methods=['POST'])
def parse_resume():
    text = request.json['text']
    preprocessed_text = preprocess_text(text)
    inputs = tokenizer(preprocessed_text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    entities = []
    for i, token_pred in enumerate(predictions[0]):
        if token_pred != 0:  # Assuming 0 is 'O' (outside of entity)
            entities.append((preprocessed_text.split()[i], token_pred.item()))
    return jsonify({'entities': entities})

if __name__ == '__main__':
    app.run(debug=True)
