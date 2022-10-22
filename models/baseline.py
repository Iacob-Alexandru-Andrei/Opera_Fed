import torch
import torch.nn as nn
from torch.nn import functional as F
# from .utils import Lambda
from models.utils import *


class LSTM(nn.Module):
    def __init__(self, feature_size, output_size):
        """
        2 layer LSTM model: feature_size --> 200 --> 3

        attr:
        seq_size: length of the sequence
        feature_size: feature size of each interval in the sequence
        """
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(feature_size,200)
        self.lstm2 = nn.LSTM(200, output_size)


    def forward(self,X):
        X, _ = self.lstm1(X)
        X, _ = self.lstm2(X)
        X = torch.flatten(X,1)
        return X


# class Encoder(nn.Module):
#     """
#     Three layer Encoder for spectrogram (1,65,501), 3 layer
#     """
#     def __init__(self,num_filters):
#         super(Encoder, self).__init__()
#         l1,l2,l3 = num_filters
#         ### 1st ###
#         self.conv1 = nn.Conv2d(1,l1,kernel_size=24,stride=1) # kernel_size=5
#         self.norm1 = nn.BatchNorm2d(l1) # nn.BatchNorm2d()
#         self.actv1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
#         ### 2nd ###
#         self.conv2 = nn.Conv2d(l1,l2,kernel_size=16,stride=2) # kernel_size=4
#         self.norm2 = nn.BatchNorm2d(l2)
#         self.actv2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
#         ### 3rd ###
#         self.conv3 = nn.Conv2d(l2,l3,kernel_size=8,stride=3) #kernel_size=3
#         self.norm3 = Lambda(lambda x:x)
#         self.actv3 = nn.Tanh()
#         self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

#     def forward(self,X):
#         X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
#         X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
#         X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
#         X = torch.flatten(X, 1)
#         # print(X.shape)
#         return X


class Encoder(nn.Module):
    def __init__(self,num_filters):
        """
        Three layers Convolutional Layers for spectrogram with size (1,65,501)
        Arguments:
        num_filters (list<int>): number of filters for each of the convolutional Layer length of list == 3
        """
        super(Encoder, self).__init__()
        assert len(num_filters) == 4
        l0,l1,l2,l3 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(l0,l1,kernel_size=5,stride=1) # kernel_size=5,     24
        self.norm1 = nn.BatchNorm2d(l1) # nn.BatchNorm2d()
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=4,stride=2) # kernel_size=4,     16
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=3,stride=3) #kernel_size=3,       8
        self.norm3 = Lambda(lambda x:x)
        self.actv3 = nn.Tanh()
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = torch.flatten(X, 1)
        # print(X.shape)
        return X
    
    
    

class Encoder_F(nn.Module):
    """
    Fourth layer Encoder for spectrogram (1,65,501), output = 1024,

    Args:
    num_filters(list)
    """
    def __init__(self,num_filters):
        super(Encoder_F, self).__init__()
        l1,l2,l3,l4 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=(5,5),stride=(2,2))
        self.norm1 = nn.BatchNorm2d(l1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=(4,4),stride=(2,2))
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=(3,3),stride=(2,2))
        self.norm3 = nn.BatchNorm2d(l3)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1,2)) # nn.AdaptiveAvgPool2d((2,2))
        ### 4th ###
        self.conv4 = nn.Conv2d(l3,l4,kernel_size=(2,2),stride=(2,2))
        self.norm4 = Lambda(lambda x:x)
        self.actv4 = nn.Tanh()
        self.pool4 = nn.AdaptiveAvgPool2d((1,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        X = torch.flatten(X, 1)

        return X

    
    
class Encoder_F(nn.Module):

    def __init__(self,num_filters):
        """
        Four layers Convolutional Layers for spectrogram with size (1,65,501)
        Arguments:
        num_filters (list<int>): number of filters for each of the convolutional Layer, length of list == 4
        """
        super(Encoder_F, self).__init__()
        l1,l2,l3,l4 = num_filters
        ### 1st ###
        self.conv1 = nn.Conv2d(1,l1,kernel_size=(5,5),stride=(2,2))
        self.norm1 = nn.BatchNorm2d(l1)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((1,2))
        ### 2nd ###
        self.conv2 = nn.Conv2d(l1,l2,kernel_size=(4,4),stride=(2,2))
        self.norm2 = nn.BatchNorm2d(l2)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((1,2))
        ### 3rd ###
        self.conv3 = nn.Conv2d(l2,l3,kernel_size=(3,3),stride=(2,2))
        self.norm3 = nn.BatchNorm2d(l3)
        self.actv3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((1,2)) # nn.AdaptiveAvgPool2d((2,2))
        ### 4th ###
        self.conv4 = nn.Conv2d(l3,l4,kernel_size=(2,2),stride=(2,2))
        self.norm4 = Lambda(lambda x:x)
        self.actv4 = nn.Tanh()
        self.pool4 = nn.AdaptiveAvgPool2d((1,2))

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.pool2(self.actv2(self.norm2(self.conv2(X))))
        X = self.pool3(self.actv3(self.norm3(self.conv3(X))))
        X = self.pool4(self.actv4(self.norm4(self.conv4(X))))
        X = torch.flatten(X, 1)

        return X    
    
    
    
class SimpleCNN(nn.Module):
    """
    Simple Conv2d NN

    parameter:
    setting (str): must be either {'1st','2nd'}
    """
    def __init__(self,setting='1st'):
        super(SimpleCNN, self).__init__()
        if setting == '1st':
            num_filter,kernel_size,latent = 32,5,238080
        elif setting == '2nd':
            num_filter,kernel_size,latent = 64,2,512000
        else:
            raise ValueError("setting must be either {'1st','2nd'}")
        ### 1st ###
        self.conv1 = nn.Conv2d(1,num_filter,kernel_size)
        self.actv_cnn = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout_cnn = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(latent,256)
        self.actv_1 = nn.ReLU()
        self.linear2 = nn.Linear(256,128)
        self.actv_2 = nn.ReLU()
        self.linear3 = nn.Linear(128,6)

    def forward(self,X):
        X = self.dropout_cnn(self.pool1(self.actv_cnn(self.conv1(X))))
        X = torch.flatten(X, 1)
        X = self.actv_1(self.linear1(X))
        X = self.actv_2(self.linear2(X))
        X = self.linear3(X)
        return X
    
    
    
    
    
    
    
class convnet(nn.Module):
    def __init__(self):

        super(convnet, self).__init__()
        #assert len(num_filters) == 4
        #l0,l1,l2,l3 = num_filters
        
        ### 1st ###
        self.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2)  
        self.norm1 = nn.BatchNorm2d(64) # nn.BatchNorm2d()
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3),stride=2)

        ########################conv block########################## 
        self.conv2 = nn.Conv2d(64,64,kernel_size=(1,1),stride=(2, 2))  
        self.norm2 = nn.BatchNorm2d(64)
        self.actv2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64,64,kernel_size=(1,1),stride=(2, 2))  
        self.norm3 = nn.BatchNorm2d(64)
        self.actv3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64,256,kernel_size=(1,1) )  
        self.norm4 = nn.BatchNorm2d(256)
        self.actv4 = nn.ReLU()  
        
        self.conv5 = nn.Conv2d(256,256,kernel_size=(1,1),stride=(2, 2) )  
        self.norm5 = nn.BatchNorm2d(256)
        self.actv5 = nn.ReLU()  
        ######################identity block#####################
      
        self.conv6 = nn.Conv2d(256,64,kernel_size=(1,1) )  
        self.norm6 = nn.BatchNorm2d(64)
        self.actv6 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(64,64,kernel_size=(3,3)  )  
        self.norm7 = nn.BatchNorm2d(64)
        self.actv7 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(64,256,kernel_size=(1,1)  )  
        self.norm8 = nn.BatchNorm2d(256)
        self.actv8 = nn.ReLU()
        ##########################################################
        self.conv9 = nn.Conv2d(256,128,kernel_size=(1,1),stride=(2, 2))  
        self.norm9 = nn.BatchNorm2d(128)
        self.actv9 = nn.ReLU()
        
        self.conv10 = nn.Conv2d(128,128,kernel_size=(1,1),stride=(2, 2))  
        self.norm10 = nn.BatchNorm2d(128)
        self.actv10 = nn.ReLU()
        
        self.conv11 = nn.Conv2d(128,512,kernel_size=(1,1) )  
        self.norm11 = nn.BatchNorm2d(512)
        self.actv11 = nn.ReLU()  
        
        self.conv12 = nn.Conv2d(512,512,kernel_size=(1,1),stride=(2, 2) )  
        self.norm12 = nn.BatchNorm2d(512)
        self.actv12 = nn.ReLU()  
        ###########################################################
        self.conv13 = nn.Conv2d(512,128,kernel_size=(1,1) )  
        self.norm13 = nn.BatchNorm2d(128)
        self.actv13 = nn.ReLU()
        
        self.conv14 = nn.Conv2d(128,128,kernel_size=(1,1) )  
        self.norm14 = nn.BatchNorm2d(128)
        self.actv14 = nn.ReLU()
        
        self.conv15 = nn.Conv2d(128,512,kernel_size=(1,1)  )  
        self.norm15 = nn.BatchNorm2d(512)
        self.actv15 = nn.ReLU()
        ###########################################################
        
        self.conv16 = nn.Conv2d(512,256,kernel_size=(1,1),stride=(2, 2))  
        self.norm16 = nn.BatchNorm2d(256)
        self.actv16 = nn.ReLU()
        
        self.conv17 = nn.Conv2d(256,256,kernel_size=(1,1),stride=(2, 2))  
        self.norm17 = nn.BatchNorm2d(256)
        self.actv17 = nn.ReLU()
        
        self.conv18 = nn.Conv2d(256,1024,kernel_size=(1,1) )  
        self.norm18 = nn.BatchNorm2d(1024)
        self.actv18 = nn.ReLU()  
        
        self.conv19 = nn.Conv2d(1024,1024,kernel_size=(1,1),stride=(2, 2) )  
        self.norm19 = nn.BatchNorm2d(1024)
        self.actv19 = nn.ReLU() 
        
        ###########################################################
        self.conv20 = nn.Conv2d(1024,256,kernel_size=(1,1) )  
        self.norm20 = nn.BatchNorm2d(256)
        self.actv20 = nn.ReLU()
        
        self.conv21 = nn.Conv2d(256,1024,kernel_size=(1,1)  )  
        self.norm21 = nn.BatchNorm2d(1024)
        self.actv21 = nn.ReLU()
        
        self.conv22 = nn.Conv2d(1024,1024,kernel_size=(1,1)  )  
        self.norm22 = nn.BatchNorm2d(1024)
        self.actv22 = nn.ReLU()
        ############################################################
        self.conv23 = nn.Conv2d(1024,512,kernel_size=(1,1),stride=(2, 2))  
        self.norm23 = nn.BatchNorm2d(512)
        self.actv23 = nn.ReLU()
        
        self.conv24 = nn.Conv2d(512,512,kernel_size=(1,1),stride=(2, 2))  
        self.norm24 = nn.BatchNorm2d(512)
        self.actv24 = nn.ReLU()
        
        self.conv25 = nn.Conv2d(512,2048,kernel_size=(1,1) )  
        self.norm25 = nn.BatchNorm2d(2048)
        self.actv25 = nn.ReLU()  
        
        self.conv26 = nn.Conv2d(2048,2048,kernel_size=(1,1),stride=(2, 2) )  
        self.norm26 = nn.BatchNorm2d(2048)
        self.actv26 = nn.ReLU() 
        ###########################################################
        self.conv27 = nn.Conv2d(2048,512,kernel_size=(1,1) )  
        self.norm27 = nn.BatchNorm2d(512)
        self.actv27 = nn.ReLU()
        
        self.conv28 = nn.Conv2d(512,512,kernel_size=(1,1) )  
        self.norm28 = nn.BatchNorm2d(512)
        self.actv28 = nn.ReLU()
        
        self.conv29 = nn.Conv2d(512,2048,kernel_size=(1,1)  )  
        self.norm29 = nn.BatchNorm2d(2048)
        self.actv29 = nn.ReLU()
        self.pool29 = nn.AdaptiveAvgPool2d((1,2))
        #############################################################

    def forward(self,X):
        X = self.pool1(self.actv1(self.norm1(self.conv1(X))))
        X = self.actv2(self.norm2(self.conv2(X)))
        X = self.actv3(self.norm3(self.conv3(X)))
        X = self.actv4(self.norm4(self.conv4(X)))
        X = self.actv5(self.norm5(self.conv5(X)))
        X = self.actv6(self.norm6(self.conv6(X)))
        X = self.actv7(self.norm7(self.conv7(X)))
        X = self.actv8(self.norm8(self.conv8(X)))
        X = self.actv9(self.norm9(self.conv9(X)))
        X = self.actv10(self.norm10(self.conv10(X)))
        X = self.actv11(self.norm11(self.conv11(X)))
        X = self.actv12(self.norm12(self.conv12(X)))
        X = self.actv13(self.norm13(self.conv13(X)))
        X = self.actv14(self.norm14(self.conv14(X)))
        X = self.actv15(self.norm15(self.conv15(X)))
        X = self.actv16(self.norm16(self.conv16(X)))
        X = self.actv17(self.norm17(self.conv17(X)))
        X = self.actv18(self.norm18(self.conv18(X)))
        X = self.actv19(self.norm19(self.conv19(X)))
        X = self.actv20(self.norm20(self.conv20(X)))
        X = self.actv21(self.norm21(self.conv21(X)))
        X = self.actv22(self.norm22(self.conv22(X)))
        X = self.actv23(self.norm23(self.conv23(X)))
        X = self.actv24(self.norm24(self.conv24(X)))
        X = self.actv25(self.norm25(self.conv25(X)))
        X = self.actv26(self.norm26(self.conv26(X)))
        X = self.actv27(self.norm27(self.conv27(X)))
        X = self.actv28(self.norm28(self.conv28(X)))
        X = self.pool29(self.actv29(self.norm29(self.conv29(X))))        
        X = torch.flatten(X, 1)
        # print(X.shape)
        return X     
    
########################################################################################################## 
    
    
class ConvNet1D4L(nn.Module):
    """
    1D Convolutional Neural Network, 4 Blocks construction
    Each Block consists: conv(l,k,s) -> batchnorm -> relu -> conv(l,k,s) -> relu
    where l,k,s are number of neurons, kernal size and stride respectively
    """
    def __init__(self,in_channels=70,**kwargs):
        """
        Args:
        in_channels (int) - input channel
        layers (list<int>) - number of neurons on each conv layer, length must be equal to 3
        kernel_sizes (list<int>) - kernel_size on each conv layer, length must be equal to 3
        strides (list<int>) - stride on on each conv layer, length must be equal to 3
        """
        super(ConvNet1D4L, self).__init__()
        # parameters
        layers = kwargs.get('layers',[128,256,512,1024])
        kernel_sizes = kwargs.get('kernel_sizes',[128,32,8,2])
        strides = kwargs.get('strides',[4,4,2,2])
        assert len(layers) == len(kernel_sizes) == len(strides)
        l0 = in_channels
        l1,l2,l3,l4 = layers
        k1,k2,k3,k4 = kernel_sizes
        s1,s2,s3,s4 = strides
        self.conv1 = nn.Conv1d(in_channels=l0,out_channels=l1,kernel_size=k1,stride=s1)
        self.conv2 = nn.MaxPool1d(kernel_size=k1,stride=s1)
        self.conv3 = nn.Conv1d(in_channels=l1,out_channels=l2,kernel_size=k2,stride=s2)
        self.conv4 = nn.MaxPool1d(kernel_size=k2,stride=s2)
        self.conv5 = nn.Conv1d(in_channels=l2,out_channels=l3,kernel_size=k3,stride=s3)
        self.conv6 = nn.MaxPool1d(kernel_size=k3,stride=s3)
        self.conv7 = nn.Conv1d(in_channels=l3,out_channels=l4,kernel_size=k4,stride=s4)
        self.conv8 = nn.MaxPool1d(kernel_size=k4,stride=s4)
        self.norm0 = nn.BatchNorm1d(l0,affine=False)
        self.norm1 = nn.BatchNorm1d(l1,affine=False)
        self.norm3 = nn.BatchNorm1d(l2,affine=False)
        self.norm5 = nn.BatchNorm1d(l3,affine=False)
        self.norm7 = nn.BatchNorm1d(l4,affine=False)
        self.flatten = nn.Flatten()

    def forward(self,X):
        X = self.norm0(X)
        X = self.conv1(X)
        X = self.norm1(X)
        X = nn.functional.relu(X)
        X = self.conv2(X)
        X = nn.functional.relu(X)
        X = self.conv3(X)
        X = self.norm3(X)
        X = nn.functional.relu(X)
        # X = self.conv4(X)
        # X = nn.functional.relu(X)
        X = self.conv5(X)
        X = self.norm5(X)
        X = nn.functional.relu(X)
        X = self.conv6(X)
        X = nn.functional.relu(X)
        X = self.conv7(X)
        X = self.norm7(X)
        X = nn.functional.relu(X)
        # X = self.conv8(X)
        # X = nn.functional.relu(X)
        X = self.flatten(X)
        return X


class ConvNet1D(nn.Module):
    """
    1D Convolutional Neural Network, 3 Blocks construction
    Each Block consists: conv(l,k,s) -> batchnorm -> relu -> conv(l,k,s) -> relu
    where l,k,s are number of neurons, kernal size and stride respectively
    """
    def __init__(self,in_channels=70,**kwargs):
        """
        Args:
        in_channels (int) - input channel
        layers (list<int>) - number of neurons on each conv layer, length must be equal to 3
        kernel_sizes (list<int>) - kernel_size on each conv layer, length must be equal to 3
        strides (list<int>) - stride on on each conv layer, length must be equal to 3
        """
        super(ConvNet1D, self).__init__()
        # parameters
        layers = kwargs.get('layers',[128,256,512])
        kernel_sizes = kwargs.get('kernel_sizes',[64,16,4])
        strides = kwargs.get('strides',[4,4,4])
        assert len(layers) == len(kernel_sizes) == len(strides)
        l0 = in_channels
        l1,l2,l3 = layers
        k1,k2,k3 = kernel_sizes
        s1,s2,s3 = strides
        self.conv1 = nn.Conv1d(in_channels=l0,out_channels=l1,kernel_size=k1,stride=s1)
        self.conv2 = nn.MaxPool1d(kernel_size=k1,stride=s1)
        self.conv3 = nn.Conv1d(in_channels=l1,out_channels=l2,kernel_size=k2,stride=s2)
        self.conv4 = nn.MaxPool1d(kernel_size=k2,stride=s2)
        self.conv5 = nn.Conv1d(in_channels=l2,out_channels=l3,kernel_size=k3,stride=s3)
        self.conv6 = nn.MaxPool1d(kernel_size=k3,stride=s3)
        self.norm0 = nn.BatchNorm1d(l0,affine=False)
        self.norm1 = nn.BatchNorm1d(l1,affine=False)
        self.norm3 = nn.BatchNorm1d(l2,affine=False)
        self.norm5 = nn.BatchNorm1d(l3,affine=False)
        self.flatten = nn.Flatten()

    def forward(self,X):
        X = self.norm0(X)
        X = self.conv1(X)
        X = self.norm1(X)
        X = nn.functional.relu(X)
        # X = self.conv2(X)
        # X = nn.functional.relu(X)
        X = self.conv3(X)
        X = self.norm3(X)
        X = nn.functional.relu(X)
        # X = self.conv4(X)
        # X = nn.functional.relu(X)
        X = self.conv5(X)
        X = self.norm5(X)
        X = nn.functional.relu(X)
        # X = self.conv6(X)
        # X = nn.functional.relu(X)
        X = self.flatten(X)
        return X
    
