from utils import divide_dataFiles, load_embedding, transform_data, TextDataset, get_time_dif
from model.textRNN import Config, textRNN
from train_val import train, evaluate,init_network
from torch.utils.data import DataLoader
import os
import time


if __name__ == "__main__":

    file_dir = 'bbc_text'

    #分割数据集，保存训练集、验证集到指定目录
    #print("Diving dataset and saving...")
    #divide_dataFiles(file_dir)
    #print("Finish dividing dataset")

    #加载词嵌入模型
    print("Loading embedding file...")
    embedding_file = os.path.join(file_dir, 'glove.6B.300d.txt')
    embedding, word_to_index = load_embedding(embedding_file)
    print("Finish loading embedding")

    #创建参数实例和模型实例
    config = Config(file_dir,embedding, word_to_index)
    model = textRNN(config).to(config.device)

    #加载数据集和迭代器
    start_time = time.time()
    print("Loading data...")
    train_file = config.train_path
    val_file = config.val_path
    classes_file = config.classes_path

    train_data = transform_data(train_file, classes_file, word_to_index, config.padding_len)
    val_data = transform_data(val_file, classes_file, word_to_index, config.padding_len)

    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    x_val, y_val = val_data[:, :-1], val_data[:, -1]

    train_dataset = TextDataset(x_train,y_train)
    val_dataset = TextDataset(x_val,y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    print("Finish loading data")
    time_dif = get_time_dif(start_time)
    print("Loading data's time usage:", time_dif)

    #train
    init_network(model)
    print(model.parameters())
    train(model, train_loader, config)
    #val
    evaluate(model, val_loader, config)


