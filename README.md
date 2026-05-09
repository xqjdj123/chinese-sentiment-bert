# 中文情感分析 | Chinese Sentiment Analysis

基于 BERT 微调的中文酒店评论情感分类模型，准确率 93%。

🔗 **在线体验：[点击这里](https://huggingface.co/spaces/xqjdj/chinese-sentiment)**

## 项目简介

输入一条中文酒店评论，模型判断其为正面或负面情感，并给出置信度。

## 技术栈

- 模型：bert-base-chinese（Hugging Face）
- 框架：PyTorch + Transformers
- 数据集：ChnSentiCorp（9600条中文酒店评论）
- 部署：Gradio + Hugging Face Spaces

## 效果展示

| 评论 | 结果 | 置信度 |
|------|------|--------|
| 房间很干净，服务态度也很好 | 正面 😊 | 99.8% |
| 空调坏了，前台态度还很差 | 负面 😞 | 99.9% |
| 这家酒店服务真的好啊，晚上热水都没有 | 正面 😊（误判） | 99.9% |

## 训练细节

- 预训练模型：bert-base-chinese
- 训练轮次：3 epochs
- 验证集准确率：93%
- 硬件：CPU

## 快速开始

```bash
pip install -r requirements.txt
python train.py        # 训练模型
python predict.py      # 本地预测
python app.py          # 启动网页界面
```

## 局限性

模型对反讽、转折句识别能力弱，例如"晚上热水都没有，完美解决了我洗热水澡的问题"会被误判为正面。
