import os
os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

# 加载数据集
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 加载tokenizer和模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# tokenize处理
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

# 训练参数
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
)

# 评估函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# 开始训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

