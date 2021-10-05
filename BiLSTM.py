'''
date: 2021/3/19
author: @流氓兔23333
content: 酒店评论数据情感分析
'''

import numpy as np
import pandas as pd
from tqdm import tqdm  
import time
import random
import torch
from sklearn.utils import shuffle as reset
import os, warnings, pickle
warnings.filterwarnings('ignore')
import sklearn.metrics as ms

import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


data_path = './data_raw/'
save_path = './temp_results/'

# 读取txt文件
# load 原文
label2id = {'neg':0, 'pos':1}

dirs_train = data_path+'data_10000/'
def read_file(dirs_train):
    '''
    return [[文章], [label]]
    '''
    data_train  = [[], []]
    for item in os.listdir(dirs_train):
        print(item, '开始读取')
        new_path = str(dirs_train + item)
        new_dirs = os.listdir(new_path)
        for f_name in tqdm(new_dirs):
            file_path = str(new_path+'/'+f_name)
            with open(file_path, encoding='utf-8') as text:
                data_train[0].append(text.read())
                data_train[1].append(label2id[item])
    data_train = np.array(data_train)
    df = pd.DataFrame({'label':data_train[1], 'review':data_train[0]})
    return df

# hotel_mark = read_file(dirs_train)
# hotel_mark.to_csv(data_path+'mark_hotel.csv')


''' 数据读取 '''
# hotel_mark = pd.read_csv(open(data_path+'mark_hotel_tag.csv', 'r'))
# hotel_mark = hotel_mark.loc[:, ['label', 'review', 'sentiment_tag']]
# hotel_mark.head()
# hotel_mark.to_csv(data_path+'ChnSentiCorp_htl_all.csv')

import transformers
from transformers import *
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.utils import shuffle as reset
import torch.nn.functional as F
import torch.optim as optim
bert_path = 'D:/competition/News_classification/bert_pretrain/'
tokenizer = BertTokenizer.from_pretrained(bert_path)

model_name = 'bert-base-chinese'
config = BertConfig.from_pretrained(model_name)


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_test_split(data_df, test_size=0.4, shuffle=True, random_state=None):
    data_len = len(data_df)
    if shuffle:
        shuffle_index = reset(range(data_len), random_state=random_state)
        data_df = data_df.iloc[shuffle_index, :]
    
    idx = int(data_len*test_size)
    data_val, data_trn = data_df.iloc[:idx].reset_index(), data_df.iloc[idx:].reset_index()
    return data_trn, data_val



class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True, bert_path=bert_path):
        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = str(self.data.loc[index, 'review'])
        

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids


class BiLSTM_Attention(nn.Module):
    def __init__(self, embed_dim, hid_dim):
        super(BiLSTM_Attention, self).__init__()
        self.hid_dim = hid_dim
        self.word_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hid_dim, bidirectional=True, batch_first=True)
        # attention, self.attn 的 out_dim 是任意的，这里直接设成 tar_dim
        self.attn = nn.Linear(self.hid_dim, 2)
        self.v = nn.Linear(2, 1, bias=False)
		
        self.fc1 = nn.Linear(self.hid_dim, self.hid_dim*3)
        self.drpout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.hid_dim*3, 2)

    def forward(self, sequence, atten_masks):
        batch_size = sequence.shape[0]
        
        # nn.Embedding 要求输入的数据为 [batch_size, seq_len]
        # 输出为 [batch_szie, seq_len, embed_dim]
        embeds = self.word_embedding(sequence)
        seq_len = embeds.shape[1]
        
        lstm_out, _ = self.lstm(embeds)
        # [bs, seq_len, hid_size]
        lstm_out = lstm_out[:, :, :self.hid_dim] + lstm_out[:, :, self.hid_dim:]


        attention = torch.zeros(seq_len, batch_size)   # [seq_len, batch_size]
        attention = attention.to(device)
        
        for t in range(seq_len):
            
            # attention_iuput [batch_size, hid_dim*2]
            # h_n[0, :, :] shape [batch_size, hid_dim]
            attention_iuput = lstm_out[:, t, :]
            # mt [batch_size, tar_size]
            mt = self.attn(attention_iuput)
            attention_t = self.v(mt)  # [batch_size, 1]
            attention[t] = attention_t.squeeze(1)
        
        attention =  attention.transpose(0,1) # [batch_size, seq_len]
        attention = attention.masked_fill(atten_masks == 0, -1e9) # mask步骤，用 -1e9 代表负无穷
        attention = F.softmax(attention, dim=1)
        
        attention = attention.unsqueeze(1) # [batch_size, 1, seq_len]
        # lstm_out [batch_size, seq_len, hid_dim]
        c = torch.bmm(attention, lstm_out) # [batch_size, 1, hid_dim]
        c = c.squeeze(1) # [batch_size, hid_dim]

        c = F.tanh(c)

        output = self.fc1(c)
        output = self.drpout(output)
        output = self.fc2(output)

        attention = attention.squeeze(1)

        return output, attention


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path+'BiLSTM_Attention.pth')
    print('The best model has been saved')


def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=2):
    loss_his = []
    try:
        checkpoint = torch.load(save_path+'BiLSTM_Attention.pth', map_location='cpu')
        checkpoint.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('-----Continue Training-----')
    except:
        print('No Pretrained model!')
        print('-----Training-----')

    model.to(device)
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch+1, epochs))
        # i, batch = next(enumerate(loader_trn))
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            output, _ = model(batch[0], batch[1])
            loss = criterion(output, batch[-1])
            if (i+1) % 8 == 0:
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()        
                loss_his.append(loss.item())
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, val_loader)


best_score = 0
def eval(model, optimizer, val_loader):
    model.eval()
    _, batch = next(enumerate(val_loader))
    # [token_ids, attn_masks, token_type_ids, label]
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        output, _ = model(batch[0], batch[1])
        label_ids = batch[-1].cpu().numpy()
        _, prediction = torch.max(F.softmax(output, dim=1), 1)
        pred_val = prediction.cpu().data.numpy().squeeze()
        acc_val = ms.accuracy_score(label_ids, pred_val)

    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)


def test(model, data_df, with_labels=False):
    checkpoint = torch.load(save_path+'BiLSTM_Attention_version2.pth', map_location='cpu')
    # checkpoint = torch.load(save_path+'BiLSTM_Attention.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print('-----Testing-----')

    # data_df = data_df.rename(columns={'评论详情': 'review'})

    print("Reading testing data...")
    dataset_tst = CustomDataset(data_df, maxlen=128, with_labels=with_labels, bert_path=bert_path)
    loader_tst = Data.DataLoader(dataset_tst, batch_size=128,  shuffle=False)
    pred_all = []
    model.eval()
    # [textid, atten_masks, text_len, doc_len, label]
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader_tst)):
            batch = tuple(t.to(device) for t in batch)
            
            output, _ = model(batch[0], batch[1])
            _, prediction = torch.max(F.softmax(output, dim=1), 1)
            pred_val = prediction.cpu().data.numpy().squeeze()
            pred_all = pred_all + list(pred_val)
            if with_labels:
                label_ids = batch[-1].cpu().numpy()
                acc_val = ms.accuracy_score(label_ids, pred_val)
                print("Validation Accuracy: {}".format(acc_val))

    # pickle.dump(atten, open(save_path+'attention_score.pkl', 'wb'))
    return pred_all


def loss_plot(loss_his):
    import matplotlib.pyplot as plt
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(100,80))
    plt.plot(loss_his, linewidth=3)
    plt.ylabel('Crossentropy Loss', size=20)
    plt.xlabel('step', size=20)
    plt.tick_params(labelsize=20)
    # plt.savefig('./loss_his.png')
    plt.show()


''' 注意力得分 '''
def atten_plot(atten, id_):
    '''
    atten 注意力得分[batch_size, seq_len]
    text_original 原词文本 [batch_size, seq_len]
    label_all [batch_size]
    id_ ：第id_个sentence
    '''
    
    atten_id = sorted(enumerate(atten[id_]), key=lambda x:x[1], reverse=True)
    text_id = tokenizer.convert_ids_to_tokens(batch[0][id_])
    
    position = [p for p,v in atten_id[:10]]
    socres = [v for p,v in atten_id[:10]]
    ax = plt.figure(figsize=(100, 80))
    ax = sns.heatmap(np.array(socres).reshape((1, len(position))), 
                xticklabels=np.array(text_id)[position], 
                yticklabels=[],
                cmap='Blues')
    ax.set_xlabel('Each Word in the Sentence', fontsize=20)
    ax.set_ylabel('Words importance', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.show()


def distribution_plot_TextLength(mark_lens):
    # 评论长度分布图：用于确定 过长文本 and 分词的 max_len
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.distplot(mark_lens, fit=stats.norm, color='g')  # 正太概率密度 / 核密度估计图
    plt.show()


if __name__ == 'main':
    
    ''' 数据读取 '''
    hotel_mark = pd.read_csv(open(data_path+'hotel_mark.csv', 'rb'), index_col=0)
    hotel_mark = hotel_mark.loc[:, ['label', 'review']]
    hotel_mark.head()
    
    mark_lens = [len(s) for s in hotel_mark['review']]
    np.argmax(mark_lens)
    hotel_mark['review'].iloc[1509]

    drop_list = ['免费注册', '网站导航', '宾客索引', '服务说明', '诚聘英才', '代理合作', '广告业务',
                 '联系我们']
    
    drop_index = []
    for i in tqdm(range(len(hotel_mark))):
        text = hotel_mark['review'].iloc[i]
        for w in drop_list:
            if w in text:
                drop_index.append(i)
                continue
    len(set(drop_index))
    
    hotel_mark_drop = hotel_mark.drop(index=hotel_mark.index[drop_index], axis=0)
    hotel_mark_drop['label'].value_counts()



    mark_lens_df = pd.DataFrame({'review_len': mark_lens})
    mark_lens_df.to_csv(save_path+'mark_lens_df.csv')
    distribution_plot_TextLength(mark_lens)
    
    # set seed
    seed=1234
    set_seed(seed)

    # train_test_split
    data_trn, data_val = train_test_split(hotel_mark, test_size=0.4, shuffle=True, random_state=seed)
    data_trn['label'].value_counts()
    data_val['label'].value_counts()

    # token_ids, attn_masks, token_type_ids, label
    batch_size = 128
    print("Reading training data...")
    dataset_trn = CustomDataset(data_trn, maxlen=128, bert_path=bert_path)
    loader_trn = Data.DataLoader(dataset_trn, batch_size=batch_size,  shuffle=True)
    print("Reading valing data...")
    dataset_val = CustomDataset(data_val, maxlen=128, bert_path=bert_path)
    loader_val = Data.DataLoader(dataset_val, batch_size=len(dataset_val),  shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device



    embed_size = 2 * int(np.floor(np.power(config.vocab_size, 0.25)))
    model = BiLSTM_Attention(embed_dim=embed_size, hid_dim=256)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    pred, atten = test(model, hotel_mark, with_labels=True)
    atten.shape
    # hotel_mark['sentiment_tag_model'] = pred
    # hotel_mark.to_csv(save_path+'hotel_mark_tag.csv')

    ms.confusion_matrix(hotel_mark['label'], pred)
    ms.recall_score(hotel_mark['label'], pred)
    ms.accuracy_score(hotel_mark['label'], pred)
    ms.precision_score(hotel_mark['label'], pred)
    ms.f1_score(hotel_mark['label'], pred)

    loss_his = pickle.load(open(save_path+'loss_his_all.pkl', 'rb'))
    loss_plot(loss_his)


    ''' 泰迪杯 '''
    scenic_mark = pd.read_excel(open('D:/competition/泰迪杯_C/data_raw/scenic_mark_tag.xlsx', 'rb'))
    scenic_mark = scenic_mark.rename(columns={'评论详情': 'review'})
    scenic_mark_new = scenic_mark.reset_index()
    scenic_mark_new = scenic_mark_new.rename(columns={'index':'id_'})
    pred = test(model, scenic_mark_new, with_labels=False)
    len(pred), len(scenic_mark)
    scenic_mark['sentiment_tag_model'] = pred
    scenic_mark = scenic_mark.rename(columns={'review':'评论详情'})
    # scenic_mark.to_excel(open('D:/competition/泰迪杯_C/data_raw/scenic_mark_tag.xlsx', 'wb'))

    hotel_mark = pd.read_excel(open('D:/competition/泰迪杯_C/data_raw/hotel_mark_tag.xlsx', 'rb'))
    hotel_mark = hotel_mark.rename(columns={'评论详情': 'review'})
    hotel_mark_new = hotel_mark.reset_index()
    hotel_mark_new = hotel_mark_new.rename(columns={'index':'id_'})
    pred = test(model, hotel_mark_new, with_labels=False)
    len(pred), len(hotel_mark)
    hotel_mark['sentiment_tag_model'] = pred
    hotel_mark = hotel_mark.rename(columns={'review': '评论详情'})
    # hotel_mark.to_excel(open('D:/competition/泰迪杯_C/data_raw/hotel_mark_tag.xlsx', 'wb'))
    

    fun_transform = lambda x: 0 if x!=1 else x
    ms.accuracy_score(hotel_mark['sentiment_tag'].apply(fun_transform), pred)
    ms.confusion_matrix(hotel_mark['sentiment_tag'].apply(fun_transform), pred)
    pred = test(model, hotel_mark, with_labels=False)
    hotel_mark['sentiment_tag_model'] = pred
    hotel_mark.to_csv(open(data_path+'hotel_mark.csv', 'w'))
    ms.accuracy_score(hotel_mark['label'], hotel_mark['sentiment_tag_model'])
    ms.accuracy_score(hotel_mark['sentiment_tag_model'], hotel_mark['sentiment_tag'])

    idx_diff = [i for i in range(len(hotel_mark)) if hotel_mark['sentiment_tag_model'].iloc[i]!=hotel_mark['sentiment_tag'].iloc[i]]
    len(idx_diff)
    