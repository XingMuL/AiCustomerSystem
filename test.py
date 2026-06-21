from modelscope.hub.snapshot_download import snapshot_download

# 下载并返回模型本地路径
model_path = snapshot_download(
    "BAAI/bge-large-zh-v1.5",
    local_dir="./bge-large-zh-v1.5",
    cache_dir=None
)
print("模型路径：", model_path)