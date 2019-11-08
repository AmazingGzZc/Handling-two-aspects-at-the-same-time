import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.utils.data as d
from tensorboardX import SummaryWriter as S
import torch.nn.functional as F
from construct_data import traingzc
from predata import DIC
from gensim.models import word2vec
from Attention import Attn
import time
writer=S(log_dir='./模型')
EPOCH=1
LR=0.01
BATCH_SIZE=20
LSTM_hidden_size=128     #LSTM隐藏层大小
Dictionary_size=266440   #字典中第一位为0，意义为padding
word_embedding_size=200
Sequence_max_length=300
word_feature_embedding=50
num_aspect=2
model=word2vec.Word2Vec.load('all_vectors.model')
word_name=['负面评价词语（中文）.txt','负面情感词语（中文）.txt','正面评价词语（中文）.txt','正面情感词语（中文）.txt']
word_list=[[],[],[],[]]
i=0
for x in word_list:
    f=open('/home/gzc/pytorh练习/multi-channel-Rcnn/hownet/'+word_name[i],'r')
    for line in f.readlines():
        x.append(line.strip('\n'+' '))
    i+=1
A=word_list[0]
B=word_list[1]
C=word_list[2]
D=word_list[3]
dic=DIC('测试评论+验证评论.txt')
w2i,i2w=dic.addsentence()
train_data=traingzc(w2i,i2w,'细粒度评论.txt','细粒度标签.txt',A,B,C,D)
'''for i in train_data:
    if i[1][1]==0:
        print(i[1][1],'456')'''
#print(train_data)
train_loader=d.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
'''test_data=testData('验证集评论.txt','验证集标签.txt')
test_loader=d.DataLoader(dataset=test_data,batch_size=500,shuffle=True)'''
w2v=np.zeros((Dictionary_size,word_embedding_size))
j=1
for i in i2w:
    if i==0:
        continue
    else:
        w2v[j]=model[i2w[i]]
        j+=1
w2v=torch.from_numpy(w2v)
class M_Aspect(nn.Module):
    def __init__(self,weight,input_size):
        super(M_Aspect,self).__init__()
        self.weight=weight
        self.input_size=input_size
        self.embedding1=nn.Embedding(Dictionary_size,word_embedding_size,padding_idx=0,_weight=self.weight)
        self.embedding2=nn.Embedding(6,word_feature_embedding,padding_idx=0)  #此处的6为只有5种词性，还有0作为padding，50为词性特征向量化大小
        self.lstm=nn.LSTM(input_size=self.input_size,
                          hidden_size=LSTM_hidden_size,
                          num_layers=2,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self.linear=nn.Linear(self.input_size+2*LSTM_hidden_size,128)#此处的128为超参数，可以自行规定。
        self.tanh=nn.Tanh()
        self.attn=Attn(128)#这里的大小根据上一层全连接层大小而来
        self.l1=nn.Linear(word_embedding_size+2*LSTM_hidden_size,64)
        self.relu1=nn.ReLU()
        self.l2=nn.Linear(64,4)
        self.relu2=nn.ReLU()
    def rep_text(self,a,b,c,d):
        a=a.view(len(b),b[0],2,LSTM_hidden_size)
        mat_FW=torch.zeros((a.shape[0],b[0],LSTM_hidden_size))
        for i in range(a.shape[0]):
            mat_FW[i][0]=d
            for l in range(1,b[i]):
                mat_FW[i][l]=a[i][l][0]
        mat_BW=torch.zeros((a.shape[0],b[0],LSTM_hidden_size))
        for j in range(a.shape[0]):
            mat_BW[j][b[j]-1]=d
            for k in range(b[j]-1):
                mat_BW[j][k]=a[j][b[j]-2-k][1]
        data=torch.cat((mat_FW,c),2)
        data=torch.cat((data,mat_BW),2)
        return data                #rep_text方法是为了获得RCNN中排序好的矩阵
    def packedd(self,x,y):
        x=x.view(-1,Sequence_max_length,self.input_size)
        #print(x.shape)
        x=x.detach().numpy()
        if torch.is_tensor(y) == True:
            y=y.numpy()
        num=[]
        for i in range(x.shape[0]):
            k=0
            for j in range(Sequence_max_length):
                if x[i][j].any()==True:
                    k+=1
            num.append(k)
        lengths=sorted(num,reverse=True)
        lengths_2=lengths
        matrix=np.zeros((x.shape[0],lengths[0],self.input_size))
        label=[]
        for i in range(x.shape[0]):
            matrix[i][:max(num)]=x[num.index(max(num))][:max(num)]
            label.append(y[num.index(max(num))])
            elment=num.index(max(num))
            num[elment]=0
        label=torch.LongTensor(label)
        #print('标签:',label)
        matrix=torch.FloatTensor(matrix)
        lengths=torch.LongTensor(lengths)
        x_packed=nn.utils.rnn.pack_padded_sequence(matrix, lengths=lengths, batch_first=True)
        return x_packed,label,lengths,matrix      #packedd方法是为了处理可变长度，x_packed是将当前batch按长度大小排好序的矩阵去经过可变长度训练，label是当前batch按照长度顺序排好之后对应的标签，lengths是一个列表，列表内的元素为当前batch内的序列长度（由大到小排列），matrix为当前batch的文本按照长度从打到小排列之后的词向量矩阵。
    def forward(self,b_x,b_y,add_feature):
        #print('b_x:',b_x,b_x.shape)
        B_X=torch.chunk(b_x,2,1)  #b_x由两部分组成，第一部分是每个字在字典中的序号，第二部分是每个字在词性字典中序号，应用chunk函数提取出每一个样本的同一行,将字典序号放在一起，将词性序号放在一起。
        #print(B_X)
        b_x=self.embedding1(B_X[0].squeeze())
        '''print(b_x,b_x.shape)
        print('---------------')'''
        if add_feature==True:
            b_x1=self.embedding1(B_X[0].squeeze())
            b_x2=self.embedding2(B_X[1].squeeze())
            b_x=torch.cat((b_x1,b_x2),2)
            #print(b_x.shape)
        x_packed,label,length,mat=self.packedd(b_x,b_y)
        b_x=b_x.view(-1,300,self.input_size)
        np.random.seed(1)
        H_0=np.random.rand(LSTM_hidden_size)#此处的H_0是为了lstm的第一个时间步都一样，人为规定。
        h_f=torch.FloatTensor(H_0)
        H_0=np.tile(H_0,(4,b_x.shape[0],1))
        H_0=torch.FloatTensor(H_0)
        C_0=H_0
        hidd=(H_0,C_0)
        r_out,(h_n,h_c)=self.lstm(x_packed,hidd)
        out = torch.nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
        out=out[0]
        data=self.rep_text(out,length,mat,h_f) #此处的data为cl+w+cr排好序的矩阵，在此处认为是记忆网络中的记忆。cl为词语左侧的隐藏层状态，w为词语的词向量，cr为词语右侧的隐藏层状态。
        datas=self.linear(data)#以下是使用自我注意力机制提取出最重要的的4个记忆
        datas=self.tanh(datas)  #先经过一层全连接层，激活函数为Tanh
        score=self.attn(datas) #此处调用已写好的注意力机制，score为返回的注意力分数。
        #print('score:',score)
        '''print('-------------')
        print(score)'''
        sen_mat=torch.zeros(num_aspect*score.shape[0],word_embedding_size+2*LSTM_hidden_size)
        #print('sen_mat:',sen_mat.shape)
        scores=score.contiguous().view(-1,score.shape[2])
        scores=scores.detach().numpy().tolist()#因为需要追踪梯度所以用detach()方法
        #scores=list(score.view(score.shape[0]*score.shape[1],score.shape[2]).numpy())
        #print('scores:',list(scores[0]))
        for i in range(sen_mat.shape[0]):    #145-154行是为了提取出最相关的4个记忆，放入sen_mat中，其中4为超参数，可以进行修改。
            max_index=[]
            inf=0
            for j in range(4):
                max_index.append(scores[i].index((max(scores[i]))))
                scores[i][scores[i].index(max(scores[i]))]=inf
            if i%2==0:
                sen_mat[i]=data[int(i/2)][max_index[0]]+data[int(i/2)][max_index[1]]+data[int(i/2)][max_index[2]]+data[int(i/2)][max_index[3]]
            else:
                sen_mat[i]=data[int((i+1)/2-1)][max_index[0]]+data[int((i+1)/2-1)][max_index[1]]+data[int((i+1)/2-1)][max_index[2]]+data[int((i+1)/2-1)][max_index[3]]
        sen_mat=self.l1(sen_mat)
        sen_mat=self.relu1(sen_mat)
        sen_mat=self.l2(sen_mat)
        sen_mat=self.relu2(sen_mat)
        return sen_mat,label.view(1,-1).squeeze()
m_a=M_Aspect(w2v,word_embedding_size)
print(m_a)
optimizer=torch.optim.SGD(m_a.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()
ACC_BEST=0
is_train=True
if is_train==True:
    for epoch in range(EPOCH):
        for step,(b_x,b_y) in enumerate(train_loader):
            output,labels=m_a(b_x,b_y,add_feature=False)
            loss=loss_func(output,labels)
            #print(loss.item())
            labels=labels.numpy()
            pred_y=torch.max(output,1)[1].data.numpy().squeeze()
            print('label:',labels)
            print('-------------')
            print(output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
