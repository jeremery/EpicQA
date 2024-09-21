import os
from openai import OpenAI
import time
import pickle
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-"
# 设置OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = ""
import numpy as np

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

prompt1 = '''
#01 你是一个农业领域问答对数据集处理专家。

#02 你的任务是根据我给出的内容，生成适合作为问答对数据集的问题，问题尽量具备多样性。

#03 问题必须贴合给定的上下文，涵盖的内容尽量丰富，尽量生成在实际农业生产中关注的问题。

#04 生成的问题必须能包含给定内容的关键信息，具有实际的价值。不要生成特别细节的问题。

#05 生成问题示例：

"""

水稻稻瘟病会引起作物什么症状？

介绍一下玉米经常遭受的病害。

水稻品种嘉早324适合在什么地区种植？

水稻品种稼禾1号适合在什么时间播种？

"""

#06 以下是我给出的内容：

"""

{{此处替换成你的内容}}

"""
'''

prompt2 = '''
#01 你是一个问答对数据集处理专家。

#02 你的任务是根据我的问题和我给出的内容，生成对应的问答对。

#03 答案要全面，多使用我的信息，内容要更丰富。

#04 你必须根据我的问答对示例格式来生成：

"""

{"query": "水稻品种6优160有哪些特征特性？", "answer": "该品种属粳型三系杂交水稻。在黄淮地区种植，全生育期156.1天，比对照豫粳6号晚熟3.1天。株高127.2厘米，穗长21.9厘米，每穗总粒数212粒，结实率74.9%，千粒重23.4克。抗性：苗瘟4级，叶瘟4级，穗颈瘟发病率3级，穗颈瘟损失率1级，综合抗性指数2.2。米质主要指标：整精米率64.9%，垩白粒率24.5%，垩白度3.0%，直链淀粉含量15.1%，胶稠度84毫米，达到国家《优质稻谷》标准3级。"}

{"query": "大麦黄花叶病症状有哪些？", "answer": "大麦黄花叶病毒在大麦上引起的典型症状是黄色花叶，在田间呈现黄色条块甚至整块麦 地呈现黄色。症状在12月下旬到翌年3月上旬出现，但是不同的大麦品种、地理环境、大麦生育期以及发病的不同病理时期，表现出的花叶程度不尽相同：发病初期于心叶上呈现淡黄绿色短条点，发病盛期新叶褪绿，上散生绿色短条点，老病叶变深黄色或橘黄色，严重的导致枯斑，在某些品种上花叶进一步发展为坏死症状，植株矮化。当温度超过 18℃时，感病大麦通常隐症。大麦和性花叶病毒可以单独或与大麦黄花叶病毒混合侵染大麦，引起的症状相似，因其在大麦品种Maris Otter上所表现的症状较轻而得名。最初的症状出现  在刚刚形成的嫩叶上，引起大小不规则的褪绿条斑，并伴随着叶片边缘向上卷曲，然后发展成花叶症状。 这种花叶有时候会导致枯斑、黄化，甚至使老叶加速死亡。和大麦黄花叶病毒一样，症状一般在早春出现，随着天气的变暖而逐渐隐症。当温度超过20℃时，新叶就不显症状。当温度处于5～10℃时，被感染的大麦生长就会迟缓，当土壤湿度很高时，这种现象就更严重。如果这种气候一直延续到4月，植株矮化的现象也就一直持续，尽管矮化的植株可以随着温度的上升而恢复正常，但是仍然会造成严重减产。病毒侵染所引起的症状表现由于不同大麦品种而有所差别，通常感病品种症状严重，抗病品种则无症 或症状轻微，六棱大麦损失较二棱大麦轻。温度是该类病害发展及症状表现的主要决定因素。"}

#05 我的问题如下：

"""

{{此处替换成你上一步生成的问题}}

"""

#06 我的内容如下：

"""

{{此处替换成你的内容}}

"""
'''

prompt = "You are a well-versed reader of this book and an expert in the field of question-answer dataset generation. Based on the following content, please generate 10 questions suitable for a question-answer dataset. The context and the question should be as concise as possible, and both the context and question must include the key information from the provided content. For these 10 questions, generate the corresponding question-answer pairs, including the context, question, correct answer, and incorrect answer. Return the results in dictionary form, with the dictionary keys being 'context', 'question', 'right_answer', and 'wrong_answer'. For a example:"
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
    # print('text_content:', len(text_content))

    # # 调用函数进行分段
    sections = split_into_sections(text_content)
    sections = sections[1:] #29
    # 打印每个段落
    responses = []
    for section in sections:
        # if len(section) ==
        res = generate_qa(prompt+example, section)
        # res_1 = generate_qa(prompt_answer+example, res)
        print(res)
        responses.append(res)
        # print(section[:100])
    np.save('qa6.npy', responses)
        # print(section[0])
        # print('---')
    
    # res = generate_qa(prompt+example, text_content)
    # np.save('qa3.0.npy', res)
    
main()
