import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import get_time_dif,save_weight
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    torch.manual_seed(123)
    for name, w in model.named_parameters():
        if exclude not in name and len(w.shape) == 2:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(model, train_loader, config):
    start_time=time.time()
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    best_loss=float('inf')
    average_loss=0.0
    best_weight=None
    epochs_list=[i+1 for i in range(config.num_epochs)]
    epoch_average_loss_list=[]
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs,targets in tqdm(train_loader,desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            inputs=inputs.to(config.device)
            targets=targets.to(config.device)
            outputs=model(inputs)
            model.zero_grad()
            loss=F.cross_entropy(outputs,targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            num_batches += 1

        average_loss = total_loss / num_batches
        epoch_average_loss_list.append(average_loss)
        if average_loss<best_loss:
            best_loss=average_loss
            best_weight=model.state_dict()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Average loss: {average_loss:.4f}, Best loss: {best_loss:.4f}")

    #保存模型权重
    last_weight=model.state_dict()
    exp_folder_path,exp_folder_name=save_weight(config.results_dir, best_weight,last_weight)
    config.current_exp_name=exp_folder_name

    time_dif=get_time_dif(start_time)
    print("Training's time usage:", time_dif)
    print(f"The model has been saved with best loss({best_loss:.4f}) and last loss({average_loss:.4f})")

    #保存训练日志
    log_path=os.path.join(exp_folder_path,"log.txt")
    with open(log_path, 'w') as file:
        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        file.write(formatted_now)
        file.write("\nTrain Log\n")
        file.write(f"Epoch: {config.num_epochs}  Batch_size: {config.batch_size}\n")
        file.write(f"learning_rate: {config.learning_rate}\n")
        file.write(f"best loss: {best_loss}  last loss: {average_loss}\n\n")
        file.write(f"other configurations:\n     sequence padding length: {config.padding_len}\n     lstm layers: {config.num_layers}\n     lstm hidden layer size: {config.hidden_size}\n     dropout: {config.dropout}\n     self-attention heads number: {config.num_heads}\n     self-attention head size:{config.head_size}")

    #绘制loss曲线和表格
    epoch_loss_image_path=os.path.join(exp_folder_path,"epoch_loss.png")
    plt.figure(figsize=(20, 10))
    plt.plot(epochs_list, epoch_average_loss_list)
    plt.xticks(epochs_list)
    plt.title('Training loss per epoch',fontsize=20)
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('loss',fontsize=18)
    plt.savefig(epoch_loss_image_path)


@torch.no_grad()
def evaluate(model, val_loader, config):
    start_time = time.time()
    model.eval()
    total_loss = 0.0
    predict_all = []
    targets_all = []
    predict_all=torch.tensor(predict_all,dtype=torch.int64).to(config.device)
    targets_all=torch.tensor(targets_all,dtype=torch.int64).to(config.device)
    for inputs, targets in val_loader:
        inputs=inputs.to(config.device)
        targets=targets.to(config.device)
        outputs=model(inputs)
        loss=F.cross_entropy(outputs,targets)
        total_loss+=loss
        predict = torch.max(outputs.data, 1)[1]
        targets_all = torch.cat((targets_all,targets),dim=0)
        predict_all = torch.cat((predict_all,predict),dim=0)

    acc = sum([1 for i in range(targets_all.shape[0]) if targets_all[i]==predict_all[i]]) / targets_all.shape[0]

    time_dif = get_time_dif(start_time)
    print("Evaluating's time usage:", time_dif)
    print(f"Accuracy:{acc:.4f}")

    #写入日志文件
    log_path=os.path.join(config.results_dir,config.current_exp_name,"log.txt")
    with open(log_path, 'r') as file:
        lines = file.readlines()

    with open(log_path, 'a') as file:
        if lines:
            file.write(f"\n\naccuracy on validation set: {acc}")
        else:
            file.write(f"\naccuracy on validation set: {acc}")
