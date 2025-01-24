# pip install --upgrade --quiet langchain sentence_transformers
# pip install --upgrade --quiet torch sentence-transformers
# pip install --quiet scikit-learn numpy
# pip install --quiet langchain_huggingface
# pip install --quiet datasets

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from sklearn.metrics import top_k_accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sentence_transformers import util
from sklearn.metrics.pairwise import cosine_similarity

from datasets import load_dataset
import time  

# 计算QQP任务指标
def evaluate_qqp(model, qqp_data):
    y_true = []
    y_pred = []

    # 遍历所有数据项
    for idx in range(len(qqp_data['question1'])):
        question1 = qqp_data['question1'][idx]
        question2 = qqp_data['question2'][idx]
        label = qqp_data['label'][idx]

        # 获取嵌入
        embeddings_q1 = model.encode([question1])
        embeddings_q2 = model.encode([question2])

        # 计算余弦相似度
        cosine_sim = cosine_similarity(embeddings_q1, embeddings_q2)[0][0]

        # 判断相似性，余弦相似度 > 0.9 认为是相似
        prediction = 1 if cosine_sim > 0.8 else 0

        # 记录实际标签和预测标签
        y_true.append(label)
        y_pred.append(prediction)

    # 计算准确率、召回率、精确率
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return acc, precision, recall

# 计算ANA任务的前1、前3准确率
def evaluate_analogy(model, analogy_data):
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0

    # 遍历所有数据项
    for idx in range(len(analogy_data['stem'])):
        stem = analogy_data['stem'][idx]
        answer = analogy_data['answer'][idx]
        choices = analogy_data['choice'][idx]

        # 获取嵌入
        embeddings_stem_1 = model.encode(stem[0])  # stem 第一个词的嵌入
        embeddings_stem_2 = model.encode(stem[1])  # stem 第二个词的嵌入

        # 计算stem两个词汇之间的相似度
        cosine_sim_stem = cosine_similarity([embeddings_stem_1], [embeddings_stem_2])[0][0]

        # 计算每个选项的相似度差值
        cosine_differences = []
        for choice in choices:
            embeddings_choice_1 = model.encode(choice[0])  # 选项第一个词的嵌入
            embeddings_choice_2 = model.encode(choice[1])  # 选项第二个词的嵌入

            # 计算当前选项两个词的相似度
            cosine_sim_choice = cosine_similarity([embeddings_choice_1], [embeddings_choice_2])[0][0]

            # 计算与stem的相似度差值
            cosine_diff = abs(cosine_sim_stem - cosine_sim_choice)
            cosine_differences.append(cosine_diff)

        # 获取按相似度差值排序后的索引
        top_indices = np.argsort(cosine_differences)

        # 判断前1、2、3、5个选项是否正确
        if top_indices[0] == answer:
            correct_1 += 1
        if top_indices[1] == answer:
            correct_2 += 1
        if top_indices[2] == answer:
            correct_3 += 1

    # 计算前1、2、3、5的准确率
    acc_1 = correct_1 / len(analogy_data['stem'])
    acc_2 = correct_2 / len(analogy_data['stem'])
    acc_3 = correct_3 / len(analogy_data['stem'])
    # acc_5 = correct_5 / len(analogy_data['stem'])

    return acc_1, acc_2, acc_3



def data_load():
  analogy_dataset = load_dataset("relbert/analogy_questions","bats")
  print(analogy_dataset['test'][:2])
  qqp_dataset = load_dataset("glue", "qqp")
  print(qqp_dataset['train'][:10])
  return analogy_dataset['test'][:500],qqp_dataset['train'][:500]



# 模型列表 其中第一个由于我的资源问题无法跑通，其他都可以
models = [
    'nvidia/NV-Embed-v2',
    'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'jinaai/jina-embeddings-v3',
    'stella-en-1.5B-v5',
    'Alibaba-NLP/gte-large-en-v1.5',
    'BAAI/bge-small-en',
    'dunzhang/stella-mrl-large-zh-v3.5-1792d',
    'Pristinenlp/alime-embedding-large-zh',
    'thenlper/gte-large-zh'
]


# 定义需要 trust_remote_code 的模型，让所有模型的 trust_remote_code 都为 true
trust_remote_code_models = {model: True for model in models}


ana,qqp=data_load()

for model_name in models:
    print(f"Evaluating model: {model_name}")
    start_time = time.time()
    trust_remote_code = trust_remote_code_models.get(model_name, False)
    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    acc_1, acc_2, acc_3 = evaluate_analogy(model, ana)
    acc, precision, recall = evaluate_qqp(model, qqp)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Analogy Task - Top 1 Accuracy: {acc_1 * 100:.2f}%")
    print(f"Analogy Task - Top 2 Accuracy: {(acc_2 + acc_1) * 100:.2f}%")
    print(f"Analogy Task - Top 3 Accuracy: {(acc_2 + acc_1 + acc_3) * 100:.2f}%")
    print(f"QQP Task - Accuracy: {acc * 100:.2f}%")
    print(f"QQP Task - Precision: {precision * 100:.2f}%")
    print(f"QQP Task - Recall: {recall * 100:.2f}%")
    print(f"Evaluation time: {elapsed_time:.2f} seconds")
    print("-" * 50)