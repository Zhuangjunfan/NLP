import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Config():
    def __init__(self, file_dir, embedding, word_to_index):
        self.model_name = "TextRNN"  #模型名字
        self.file_dir = file_dir  #所有数据文件根目录
        self.results_dir = "results"
        self.current_exp_name=None
        self.train_path = os.path.join(file_dir, "train.csv")  #训练集文件路径
        self.val_path = os.path.join(file_dir, "val.csv")  #验证集文件路径
        self.classes_path = os.path.join(file_dir, "classes.txt")  #类别文件路径
        self.num_classes = 5  #类别数
        self.embedding_pretrained = embedding  #预训练词嵌入模型
        self.word_to_index = word_to_index  #词汇索引字典
        self.embedding_dim = embedding.shape[1]  #词嵌入维度
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设备

        self.padding_len = 400  #文本平均词数
        self.batch_size = 32  #批次大小
        self.num_epochs = 40  #epoch大小
        self.learning_rate = 1e-5  #学习率
        self.hidden_size = 1024  #隐藏层维度
        self.num_layers = 2  #lstm层数
        self.dropout = 0.5  #随机失活
        self.num_heads=10  #注意力头数量
        self.head_size=self.embedding_dim//self.num_heads  #注意力头维度

class textRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.MultiHeadAttention=MultiHeadAttention(config)
        self.lstm = nn.LSTM(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,num_layers=config.num_layers,
                          bidirectional=True, #双向lstm，捕获文本上文和下文信息，输出维度为单向lstm的输出维度的两倍
                          batch_first=True, dropout=config.dropout
                          )
        self.fc = nn.Linear(config.hidden_size * 2,config.num_classes)
        self.ln1=nn.LayerNorm(config.embedding_dim)
        self.ln2=nn.LayerNorm(config.embedding_dim)

    def forward(self,x): #(B,T)
        x = self.embedding(x) #(B,T,embedding_dim)
        x = x + self.MultiHeadAttention(self.ln1(x))  # (B,T,embedding_dim)
        x, _ = self.lstm(self.ln2(x)) #(B,T,hidden_size * 2)
        B,T,C = x.shape
        x = self.fc(x[:,-1,:])
        return x

class Head(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.key=nn.Linear(config.embedding_dim,config.head_size,bias=False)
        self.query=nn.Linear(config.embedding_dim,config.head_size,bias=False)
        self.value=nn.Linear(config.embedding_dim,config.head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.padding_len, config.padding_len)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        B,T,C=x.shape
        k = self.key(x)  #(B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.head_size * config.num_heads, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #(B,T,head_size * num_heads)
        out = self.dropout(self.proj(out)) #(B,T,embedding_dim)
        return out
