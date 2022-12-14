import torch
import torch.nn as nn
from torch.nn import functional as F

class Flatten(nn.Module):

    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, input):
        return torch.flatten(input,start_dim=self.start_dim)

class Stack(nn.Module):
    def forward(self, x):
        return torch.cat((x,x,x),axis=1)

class Lambda(nn.Module):
    """
    transform tensor according to function func
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ED_module(nn.Module):
    """
    Combine Encoder and Decoder to form the model
    """
    def __init__(self, encoder, decoder):
        super(ED_module, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return X


class Classifier(nn.Module):
    """
    Linear Classifier (Double Layer)

    Hint: user nn.Linear as Classifier (Single Layer)
    """
    def __init__(self,input_shape,hidden_shape,output_shape):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(input_shape,hidden_shape)
        self.linear2 = nn.Linear(hidden_shape,output_shape)

    def forward(self,X):
        X = F.leaky_relu(self.linear1(X))
        X = F.dropout(X,0.1)
        X = self.linear2(X)
        return X

    
#     inputs = keras.Input(shape=input_shape)
#     features = encoder(inputs)
#     features = layers.Dropout(0.5)(features)
#     features = layers.Dense(512, activation="relu")(features)
#     features = layers.Dropout(0.5)(features)
#     outputs = layers.Dense(6, activation="softmax")(features)    
    
    
    
    
    
    
    
    