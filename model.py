import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxx"
# 设置OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = ""
from openai import OpenAI

client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion)  # 响应
print(completion.choices[0].message)  # 回答
