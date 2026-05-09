from transformers import pipeline

classifier = pipeline("sentiment-analysis",
                      model="uer/roberta-base-finetuned-jd-binary-chinese")

result = classifier("这家餐厅的菜真的太好吃了，下次还会来！")
print(result)