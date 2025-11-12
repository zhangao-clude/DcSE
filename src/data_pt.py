import numpy as np
import pandas as pd
import itertools
import torch
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset, DataLoader


# 获取Tokenizer，创建一个基于ACGT的词汇表
def get_tokenizer():
    f = ['a', 'c', 'g', 't']
    c = itertools.product(f, f, f, f, f, f)
    res = []
    for i in c:
        temp = i[0] + i[1] + i[2] + i[3] + i[4] + i[5]
        res.append(temp)
    res = np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer


# 将DNA序列转换为6-mer词汇序列
def sentence2word(str_set):
    word_seq = []
    for sr in str_set:
        tmp = []
        for i in range(len(sr) - 5):
            if 'N' in sr[i:i + 6]:
                tmp.append('null')
            else:
                tmp.append(sr[i:i + 6])
        word_seq.append(' '.join(tmp))
    return word_seq


# 将词汇序列转换为数字序列
def word2num(wordseq, tokenizer, MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return numseq


# 将DNA序列转换为数字序列
def sentence2num(str_set, tokenizer, MAX_LEN):
    wordseq = sentence2word(str_set)
    numseq = word2num(wordseq, tokenizer, MAX_LEN)
    return numseq


# 保存数据集为pt文件
def save_datasets(train_dataset, test_dataset, filename="datasets.pt"):
    # 从Dataset中提取序列和标签
    train_sequences = [seq for seq, _ in train_dataset]
    train_labels = [label for _, label in train_dataset]

    test_sequences = [seq for seq, _ in test_dataset]
    test_labels = [label for _, label in test_dataset]

    # 将序列和标签转换为PyTorch的张量
    train_sequences_tensor = torch.stack(train_sequences)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)

    test_sequences_tensor = torch.stack(test_sequences)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

    # 将张量保存为.pt文件
    torch.save({
        'train_sequences': train_sequences_tensor,
        'train_labels': train_labels_tensor,
        'test_sequences': test_sequences_tensor,
        'test_labels': test_labels_tensor
    }, filename)
    print(f"Datasets saved to {filename}")


# 从pt文件加载数据集
def load_datasets(filename="datasets.pt"):
    data = torch.load(filename, weights_only=False)

    train_sequences_tensor = data['train_sequences']
    train_labels_tensor = data['train_labels']
    test_sequences_tensor = data['test_sequences']
    test_labels_tensor = data['test_labels']

    # 创建TensorDataset
    train_dataset = TensorDataset(train_sequences_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_labels_tensor)

    # # 使用DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Loaded data from {filename}")
    return train_dataset, test_dataset


# 加载训练和测试数据集
# def load_data(species):
#     for i in range(10):
#         train_file = f'./data/datasets/train_{species}_split_{i}.csv'
#         test_file = f'./data/datasets/test_{species}_splist_{i}.csv'
#
#         train_csv = pd.read_csv(train_file)
#         test_csv = pd.read_csv(test_file)
#
#         train_labels = train_csv['label'].tolist()
#         test_labels = test_csv['label'].tolist()
#         train_sequences = train_csv['sequence'].tolist()
#         test_sequences = test_csv['sequence'].tolist()
#
#         tokenizer = get_tokenizer()
#
#         # 转换为数字序列并创建TensorDataset
#         train_sequences_num = sentence2num(train_sequences, tokenizer, 3000)
#         test_sequences_num = sentence2num(test_sequences, tokenizer, 3000)
#
#         # 标准化
#         scaler = StandardScaler()
#         train_sequences_scaled = scaler.fit_transform(train_sequences_num)  # 仅在训练集拟合
#         test_sequences_scaled = scaler.transform(test_sequences_num)  # 测试集用相同参数
#
#         # 创建 TensorDataset
#         train_dataset = TensorDataset(
#             torch.tensor(train_sequences_scaled, dtype=torch.float32),
#             torch.tensor(train_labels, dtype=torch.float32)
#         )
#         test_dataset = TensorDataset(
#             torch.tensor(test_sequences_scaled, dtype=torch.float32),
#             torch.tensor(test_labels, dtype=torch.float32)
#         )
#         save_datasets(train_dataset, test_dataset, f"./data/datasets/datasets_{species}_split_{i}.pt")
#     # return train_dataset, test_dataset

def load_data(species):
    for i in range(10):
        train_file = f'./data/datasets/train_{species}_split_{i}.csv'
        test_file = f'./data/datasets/test_{species}_split_{i}.csv'

        train_csv = pd.read_csv(train_file)
        test_csv = pd.read_csv(test_file)

        train_labels = train_csv['label'].tolist()
        test_labels = test_csv['label'].tolist()
        train_sequences = train_csv['sequence'].tolist()
        test_sequences = test_csv['sequence'].tolist()

        tokenizer = get_tokenizer()

        # 转换为数字序列并创建TensorDataset
        train_sequences_num = sentence2num(train_sequences, tokenizer, 3000)
        test_sequences_num = sentence2num(test_sequences, tokenizer, 3000)

        train_dataset = TensorDataset(torch.tensor(train_sequences_num, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(test_sequences_num, dtype=torch.float32),
                                     torch.tensor(test_labels, dtype=torch.float32))
        save_datasets(train_dataset, test_dataset, f"./data/datasets/datasets_{species}_split_{i}.pt")


# return train_dataset, test_dataset


if __name__ == '__main__':
    species = 'human'
    load_data(f"{species}")
    # # 加载数据
    # train_dataset, test_dataset = load_data(f"{species}")
    # print(train_dataset, test_dataset)
    #
    # # 保存数据集为.pt文件
    # save_datasets(train_dataset, test_dataset, f"datasets_{species}_new.pt")
    #
    # # 加载数据集
    # train_loader, test_loader = load_datasets(f"datasets_{species}_new.pt")
    #
    # # 示例：迭代训练数据
    # for batch_idx, (sequences, labels) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}, Sequences: {sequences.shape}, Labels: {labels.shape}")
