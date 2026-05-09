import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained("./model/checkpoint-1800")
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = probs.argmax().item()
    confidence = probs.max().item()
    label = "正面 😊" if pred == 1 else "负面 😞"
    return f"{label}（置信度：{confidence:.1%}）"

# 界面
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="输入一条中文评论..."),
    outputs=gr.Textbox(label="情感分析结果"),
    title="中文情感分析",
    description="基于BERT微调，输入酒店/餐厅评论，判断正面还是负面情感。",
    examples=[
        ["房间很干净，服务态度也很好，下次还会来"],
        ["空调坏了，前台态度还很差，非常失望"],
        ["这家酒店服务真的好啊，晚上热水都没有，完美解决了我洗热水澡的问题"],
    ]
)

demo.launch()