from datasets import load_dataset

# 直接加载数据集，自动缓存，不用手动下载文件
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 打印一条数据看看
print(dataset["train"][0])