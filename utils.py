import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import os
import time
from datetime import timedelta
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

def divide_dataFiles(file_dir):
    """分割并保存训练集、验证集文件，保存所有类别"""
    data=pd.read_csv(os.path.join(file_dir,'bbc_text_cls.csv'))
    # 分离特征和标签
    x = data['text']
    y = data['labels']

    # 分离出5个类别的数据
    business = x[y == 'business']
    sport = x[y == 'sport']
    politic = x[y == 'politics']
    entertainment = x[y == 'entertainment']
    tech = x[y == 'tech']

    # 打乱数据
    business = business.sample(frac=1, random_state=123).reset_index(drop=True)
    sport = sport.sample(frac=1, random_state=234).reset_index(drop=True)
    politic = politic.sample(frac=1, random_state=345).reset_index(drop=True)
    entertainment = entertainment.sample(frac=1, random_state=456).reset_index(drop=True)
    tech = tech.sample(frac=1, random_state=567).reset_index(drop=True)

    #拼接各个分类数据的文本和标签，分出训练集、验证集
    x_train_all = pd.concat([business, sport, politic, entertainment, tech])
    y_train_all = pd.concat([pd.Series(['business' for i in range(len(business))]),
                             pd.Series(['sport' for i in range(len(sport))]),
                             pd.Series(['politics' for i in range(len(politic))]),
                             pd.Series(['entertainment' for i in range(len(entertainment))]),
                             pd.Series(['tech' for i in range(len(tech))]), ])

    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, stratify=y_train_all,random_state=789)

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train = pd.DataFrame({'text': x_train, 'labels': y_train})
    test = pd.DataFrame({'text': x_test, 'labels': y_test})

    #保存训练集和验证集
    train.to_csv(os.path.join(file_dir,'train.csv'), index=False)
    test.to_csv(os.path.join(file_dir,'val.csv'), index=False)
    #找到所有类别
    classes=data['labels'].unique()
    classes=pd.Series(classes)
    classes.to_csv(os.path.join(file_dir,'classes.txt'),index=False,header=False)

def load_embedding(path):
    """加载词嵌入模型，创建词汇索引字典"""
    embedding=[]
    vocab_dic={}
    index=0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            values=line.strip().split()
            word=values[0]
            vector=values[1:]
            vector=[float(num) for num in vector]
            embedding.append(vector)
            vocab_dic[word]=index
            index=index+1
    embedding=torch.tensor(embedding)

    #在embedding末尾加一行对未知词的随机张量
    torch.manual_seed(123)
    row,column=embedding.shape
    UNK_wordvector=torch.randint(100,(1,column))
    embedding=torch.cat((embedding,UNK_wordvector), dim=0)
    vocab_dic['<UNK>']=row
    return embedding,vocab_dic

def transform_data(file_path,classes_path,word_to_index,padding_len):
    """根据词汇索引字典转换数据"""
    #nltk.download()
    data=pd.read_csv(file_path).to_numpy()
    classes=np.loadtxt(classes_path,dtype=str)
    texts=data[:,0]
    labels=data[:,1]
    tokenized_texts=[word_tokenize(t.lower()) for t in texts]
    for i in range(len(labels)):
        if (labels[i] == classes[0]): labels[i] = 0
        if (labels[i] == classes[1]): labels[i] = 1
        if (labels[i] == classes[2]): labels[i] = 2
        if (labels[i] == classes[3]): labels[i] = 3
        if (labels[i] == classes[4]): labels[i] = 4
    labels = np.array(labels)
    labels = labels[:, np.newaxis].astype(np.int64)
    #对文本序列中的所有词转换成词汇表的索引
    for row,text in enumerate(tokenized_texts):
        indexed_text = []
        for vocab in text:
            if vocab in word_to_index:
                indexed_text.append(word_to_index[vocab])
            else:
                indexed_text.append(word_to_index['<UNK>'])
        tokenized_texts[row] = indexed_text

    #将所有文本序列填充或截断成固定长度
    processed_texts = [
        text[:padding_len] if len(text) > padding_len else text + [0] * (padding_len - len(text))
        for text in tokenized_texts
    ]
    tokenized_texts = np.array(processed_texts).astype(np.int64)
    #标签和特征重新构成数据集
    transformed_data = np.concatenate((tokenized_texts, labels), axis=1)
    transformed_data = torch.from_numpy(transformed_data)
    return transformed_data

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

def save_weight(dir,best_weight,last_weight):
    count=1
    exp_folder_name = f"exp{count}"
    exp_folder_path = os.path.join(dir, exp_folder_name)
    while(os.path.isdir(exp_folder_path)):
        count+=1
        exp_folder_name = f"exp{count}"
        exp_folder_path = os.path.join(dir, exp_folder_name)
    os.mkdir(exp_folder_path)
    weights_folder_path=os.path.join(exp_folder_path,"weights")
    os.mkdir(weights_folder_path)
    torch.save(best_weight,os.path.join(weights_folder_path,"best.pt"))
    torch.save(last_weight, os.path.join(weights_folder_path,"last.pt"))
    return exp_folder_path,exp_folder_name