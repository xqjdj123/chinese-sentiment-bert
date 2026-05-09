from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载训练好的模型
model_path = "./results/checkpoint-1800" # 如果报错就换成 "./results"
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 预测函数
def predict(text):
    inputs = tokenizer(text, return_tensors="pt",
                      truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = probs.argmax().item()
    confidence = probs.max().item()
    label = "正面 😊" if pred == 1 else "负面 😞"
    return f"{label}（置信度：{confidence:.1%}）"

# 测试几句话
tests = [
    "房间很干净，服务态度也很好，下次还会来",
    "空调坏了，前台态度还很差，非常失望",
    "位置不错，就是价格有点贵",
    "这家酒店服务真的好啊，晚上热水都没有，完美解决了我洗热水澡的问题",  # 你Day3举的反讽例子
]

for t in tests:
    print(f"评论：{t}")
    print(f"结果：{predict(t)}\n")