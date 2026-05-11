from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import json

# -----------------------
# DATA LOAD
# -----------------------
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# -----------------------
# LABEL MAP
# -----------------------
label_list = ["O", "KISI", "YER", "KURUM"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# -----------------------
# MODEL
# -----------------------
model_name = "savasy/bert-base-turkish-ner-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def encode(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    word_ids = tokenized.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[example["ner_tags"][word_idx]])
        else:
            label_ids.append(-100)

        previous_word_idx = word_idx

    tokenized["labels"] = label_ids
    return tokenized

tokenized = dataset.map(encode)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# -----------------------
# TRAINING
# -----------------------
training_args = TrainingArguments(
    output_dir="./ner_model",
    num_train_epochs=8,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    logging_steps=1,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train()

model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

print("Model başarıyla eğitildi!")