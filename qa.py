import os
from openai import OpenAI
import time
import pickle

# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxx"
# 设置OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = ""
import numpy as np

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)


prompt = "You are a well-versed reader of this book and an expert in the field of question-answer dataset generation. Based on the following content, please generate 10 questions suitable for a question-answer dataset. The context and the question should be as concise as possible, and both the context and question must include the key information from the provided content. For these 10 questions, generate the corresponding question-answer pairs, including the context, question, correct answer, and incorrect answer. Return the results in dictionary form, with the dictionary keys being 'context', 'question', 'right_answer', and 'wrong_answer'. An example would be:"
prompt_answer = 'Please help me convert the given Q&A pairs into the following format:'
example = '{ "context": "Drona protected the Kaurava Vahinis for five days.", "question": "For how many days did Drona protect the Kaurava Vahinis?", "right_answer": "Five days.", "wrong_answer": "Ten days." }'

def generate_question(text_content):  # 生成问题
    content = "生成8个合适作为问答对的问题"
    prompt = prompt1.replace("{{此处替换成你的内容}}", text_content)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    )
    start_time = time.time()
    print("耗时", time.time() - start_time)
    print(completion.choices[0].message)  # 回答
    # print(completion.choices[0].message.content)
    queries_content = completion.choices[0].message.content
    queries = queries_content.strip().split('\n')
    return queries


def generation_answer(text_content, question):
    prompt = prompt2.replace("{{此处替换成你上一步生成的问题}}", question).replace("{{此处替换成你的内容}}", text_content)
    user_content = "根据问题和对应的上下文生成答案，并以问答的形式输出结果。"
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": 'user', "content": user_content}
        ]
    )
    start_time = time.time()
    print("耗时", time.time() - start_time)
    # print(completion.choices[0].message)  # 问答对
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def generate_qa(prompt ,text_content):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": 'user', "content": text_content}
        ]
    )
    start_time = time.time()
    print("耗时", time.time() - start_time)
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def read_file(file_name):
    try:
        with open(file_name, "r", encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")

def write_to_file(file_name, content):
    with open(file_name, 'wb') as f:
        pickle.dump(content, f)

def split_into_sections(text):
    """
    将输入文本按关键词 "Section " 进行分段。

    参数:
    text (str): 输入的长文本字符串。

    返回:
    sections (list): 一个包含分段的列表，每个元素是一个段落字符串。
    """
    # 使用 "Section " 作为分隔符将文本进行分割
    sections = text.split('Section ')

    # 去除空白部分，并重新加上 "Section " 前缀，除了第一个段落
    sections = [f"Section {section.strip()}" if i > 0 else section.strip() for i, section in enumerate(sections)]

    # 返回包含所有段落的列表
    return sections


def main():
    QA_list = []
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path))
    file_path = os.path.join(father_path,  'Mahabharata.txt')
    save_path = os.path.join(father_path,  'qa.json')
    text_content = read_file(file_path)
    print('text_content:', len(text_content))

    # # 调用函数进行分段
    sections = split_into_sections(text_content)
    sections = sections[1:] #29
    # 打印每个段落
    responses = []
    for section in sections:
        # if len(section) ==
        res = generate_qa(prompt+example, section)
        res_1 = generate_qa(prompt_answer+example, res)
        print(res_1)
        responses.append(res_1)
        # print(section[:100])
    np.save('qa3.0.npy', responses)
        # print(section[0])
        # print('---')
    
    # res = generate_qa(prompt+example, text_content)
    # np.save('qa3.0.npy', res)
    
main()
