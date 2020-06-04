import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sklearn
import sklearn.model_selection
from torch.autograd import Variable
from fastprogress import master_bar, progress_bar
import imblearn
from imblearn import over_sampling
import math as math
import copy as copy

class CustomDataLoader:

    def __init__(self,dataframe,window_size=4,drop_rate=False):
        self.df = dataframe.copy().drop(['date'],axis=1)
        self.X = []
        self.y = []
        second_copy = None
        if(drop_rate):
          second_copy = self.df.copy()
          self.df = self.df.drop(['rate'],axis=1)

        self.window_size = window_size

        for i in range(int(len(self.df)/5)):
            Xdat = self.df[i*5:i*5+self.window_size].values.tolist()
            ydat = None
            if drop_rate:
              ydat =second_copy.rate[(i+1)*5 - 1]
            else:
              ydat = self.df.rate[(i+1)*5 - 1]

            self.X.append(Xdat)
            self.y.append(ydat)
    
    def batchData(self,window_size=4,batch_size=32):
        no_of_batches = math.ceil(len(X)/32.0)
        dataX = []
        dataY = []
        for i in range(no_of_batches):
            datX,datY=None,None
            if (i+1)*32 > len(y):
                datX = self.X[i*32:len(y)]
                datY = self.y[i*32:len(y)]
            else:
                datX = self.X[i*32:(i+1)*32]
                datY = self.y[i*32:(i+1)*32]

            dataX.append(datX)
            dataY.append(datY)
        
        return dataX,dataY

    def getData(self):
        return (self.X,self.y)

    def resampledData(self,batch=False):
        X = copy.deepcopy(self.X)
        y = copy.deepcopy(self.y)

        index_1 = []
        index_0 = []
        for i in range(len(y)):
            if int(y[i])==1:
                index_1.append(i)
            else:
                index_0.append(i)

        no_of_0,no_of_1 = len(index_0),len(index_1)

        print(no_of_0,no_of_1)
        too_much_0 = (no_of_0>no_of_1)
        
        if too_much_0: # Increase the number of 1
            increase_ratio = float(no_of_0/no_of_1)
            add_num = len(index_1)*increase_ratio - len(index_1)
            index = 0
            for i in range(int(add_num)):
                y.append(1)
                X.append(X[index_1[index]])
                index +=1
                index %= len(index_1)
        else:   # Increase the number of 0
            increase_ratio = float(no_of_1/no_of_0 )
            add_num = len(index_0)*increase_ratio - len(index_0)

            for i in range(int(add_num)):
                y.append(0)
                X.append(X[index_0[i]])
                index +=1
                index %= len(index_0)

        return X,y

def generate_dict(X,y,device,test=0.2):
    X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=test)

    train_X,test_X,train_y,test_y = None,None,None,None
    train_X = Variable(torch.Tensor(X_train))
    train_y = Variable(torch.Tensor(np.array(y_train)))
    train_X = torch.unsqueeze(train_X, 2).to(device)
    train_y = torch.unsqueeze(train_y, 1).to(device)

    test_X = Variable(torch.Tensor(X_test))
    test_y = Variable(torch.Tensor(np.array(y_test)))
    test_X = torch.unsqueeze(test_X, 2).to(device)
    test_y = torch.unsqueeze(test_y, 1).to(device)
    
    return {'train_X':train_X,'test_X':test_X,'train_y':train_y,'test_y':test_y}

    
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,device):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        #self.seq_length = seq_length
        self.dropout = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers,dropout = 0.25)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device))
        x = x.view(1,4,self.input_size)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        out = self.dropout(out)
       
        return out[-1,]


class Trainable:

    ### Pass in ght data_dict and model
    def __init__(self, data_dict,device,model=None,loss=None,optim = None,schedule=None,hs=16,n_layers=None,n_features=None,lr=None):
        self.data_dict = data_dict
        self.device = device
        ## default params
        num_epochs = 10
        self.learning_rate = 1e-3
        if lr is not None:
          self.learning_rate = lr
        input_size = 13
        if n_features is not None:
          input_size = n_features
        #print("Input_Size:",input_size)
        hidden_size = hs
        num_layers = 1
        num_classes = 1
        if n_layers is not None:
            num_layers = n_layers
        ################
        self.model = LSTM(num_classes, input_size, hidden_size, num_layers,self.device)
        self.criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)

        if loss is not None:
            self.criterion = loss
        if optim is not None:
            self.optimizer=optim
        if schedule is not None:
            self.scheduler = schedule
        if model is not None:
            self.model = model

        

    def train(self,n_epochs=10):
        train_X = self.data_dict['train_X']
        test_X =  self.data_dict['test_X']
        train_y = self.data_dict['train_y']
        test_y =  self.data_dict['test_y']

        #####  Parameters  ######################


        #####Init the Model #######################
        #self.model = LSTM(num_classes, input_size, hidden_size, num_layers)
        self.model.to(self.device)

        ##### Set Criterion Optimzer and scheduler ####################
        criterion = torch.nn.MSELoss().to(self.device)    # mean-squared error for regression
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=500,factor =0.5 ,min_lr=1e-7, eps=1e-08)
        #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

        # Train the model
        running_loss = 0
        running_num = 0
        for epoch in progress_bar(range(n_epochs)): 
            self.model.train()
            for i in range(len(train_X)):
                input_x = train_X[i]
                input_y = train_y[i]

                outputs = self.model(input_x.to(self.device))

                optimizer.zero_grad()
            
                # obtain the loss function
                loss = self.criterion(outputs, input_y.to(self.device))
                running_loss += loss
                running_num += 1
                loss.backward()
                
                optimizer.step()
                
            if epoch % 1 == 0:
                loss_val = running_loss/float(running_num)
                print("Epoch: %d, loss: %1.5f" %(epoch, loss_val))
                running_loss=0
                running_num=0
        return self.model
    
    def test_printout(self):
        test_X = self.data_dict['test_X']
        test_y = self.data_dict['test_y']

        running_corrects = 0
        running_num = 0
        torch.set_printoptions(edgeitems=len(test_X))
        self.model.eval()
        preds=[]
        for i in range(len(test_X)):
            test_predict = self.model(test_X[i].to(self.device))
            score = 0
            if test_predict>0.5:
                score=1
            else:
                score = 0
            preds.append(score)

            if score==test_y[i]:
                running_corrects+=1

            running_num+=1

        #print(sklearn.metrics.confusion_matrix(test_y.cpu(),preds))
        accuracy = running_corrects/float(running_num)
        #print("accuracy: ", accuracy)
        #print("Report\n",sklearn.metrics.classification_report(test_y.cpu(),preds))
        return accuracy