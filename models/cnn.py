import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, vgg16, alexnet
from functools import partial
from models.utils import Flatten
from models.baseline import Encoder 
from models.baseline import convnet


from typing import Callable


class ToyCNN(nn.Module):
    def __init__(self, input_channels: int, output_dim: int):
        """Toy fully connected model

        Args:
            output_dim (int): number of classes
        """
        super(ToyCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(234896, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def toy_model_generator(input_channels: int, output_dim: int) -> Callable[[], ToyCNN]:
    """Wrapper for call from hydra

    Args:
        input_dim (int):
        output_dim (int):
    """

    def generate_toy_model():
        return ToyCNN(input_channels=input_channels, output_dim=output_dim)

    return generate_toy_model



def resnet_finetune(model, n_classes):
    """
    This function prepares resnet to be finetuned by:
    1) freeze the model weights
    2) cut-off the last layer and replace with a new one with the correct classes number
    """
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, n_classes)
    return model


class Attention(nn.Module):
    def __init__(self,feature_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(feature_size,1)

    def forward(self,X):
        assert len(X.shape) == 3
        a = self.linear(X)
        a = torch.relu(a)
        a = F.softmax(a,dim=1)
        return a*X

def create_baseline_encoder(num_filters=[1,32,64,96],scale_factor=1):
    model = Encoder(num_filters=num_filters)
    model = torch.nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=scale_factor),
          model,Flatten())
    return model   
    
# def create_baseline_encoder(num_filters=[32,64,96],scale_factor=1):
#     """
#     VGG 16 for 1 channel image, output: 512*output_size
#     """
#     model = Encoder(num_filters=num_filters)
#     model = torch.nn.Sequential(nn.UpsamplingNearest2d(scale_factor=scale_factor),
#                                 model,Flatten())
#     return model

def create_convnet(scale_factor=1):
    """
    VGG 16 for 1 channel image, output: 512*output_size
    """
    model = convnet()
    model = torch.nn.Sequential(nn.UpsamplingNearest2d(scale_factor=scale_factor),
                                model,Flatten())
    return model  


def create_alexnet(output_size=(2,2),scale_factor=1):  # output_size=(2,2)
    """Return net and out_size(512*w*l)"""
    net = alexnet()
    net = torch.nn.Sequential(Stack(),
                              nn.UpsamplingNearest2d(scale_factor=scale_factor),
                              *(list(net.children())[0:-2]),
                              nn.AdaptiveAvgPool2d(output_size),
                              Flatten())
    
#     new_features = list(net.features.children())
#     new_features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     net = nn.Sequential(
#         *new_features, 
#         nn.Sequential( nn.AdaptiveAvgPool2d(output_size) , nn.Flatten() )    # *(list(net.children())[1:-1] )
#         )    
    
    
    out_size = 256*output_size[0]*output_size[1]
    return net, out_size

def create_resnet18(output_size=(2,2)):
    """Return net and out_size(512*w*l)"""
    net = resnet18()
    net = torch.nn.Sequential(Stack(),
                              *(list(net.children())[:-2]),
                              nn.AdaptiveAvgPool2d(output_size),
                              Flatten())

#     net = nn.Sequential(
#                  nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),# customized input layer chong
#                  nn.Sequential(*(list(resnet.children())[1:-2]) ,
#                  nn.AdaptiveAvgPool2d(output_size),
#                  nn.Flatten()) 
#                     )
    
    out_size = 512*output_size[0]*output_size[1]
    return net, out_size



def create_vgg16(output_size=(2,2)):
    """
    VGG 16 for 1 channel image, output: 512*output_size
    """
    net = vgg16()
    net = torch.nn.Sequential(Stack(),
                                *(list(net.children())[:-2]),
                                nn.AdaptiveAvgPool2d(output_size),
                                Flatten())

#     new_features = list(net.features.children())
#     new_features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#     net = nn.Sequential(
#         *new_features, 
#         nn.Sequential(nn.AdaptiveAvgPool2d(output_size) , 
#         nn.Flatten())  
#         )

    out_size = 512*output_size[0]*output_size[1]
    return net, out_size


def create_vgg16_atn(output_size=(2,2)):
    """
    VGG 16 for 1 channel image, with linear attention, output: 512*output_size
    """
    mdl = vgg16()
    model = torch.nn.Sequential(Stack(),
                                *(list(mdl.children())[:-2]),
                                nn.AdaptiveAvgPool2d(output_size),
                                Flatten(2),
                                Attention(output_size[0]*output_size[1]),
                                Flatten(1)
                                )
    return model

# resnet18 = partial(resnet_finetune, resnet18(pretrained=True))


class Residual_Block(nn.Module):
    """
    Single channel for radio image (30,30,1)
    """
    def __init__(self,in_feature,out_feature,kernel_size,stride, padding='zeros', bias=False):
        super(Residual_Block, self).__init__()
        ### 1st ###
        self.conv1 = nn.Conv2d(in_feature,out_feature,kernel_size=kernel_size,stride=stride, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_feature)
        self.actv1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=kernel_size,stride=stride)
        self.covr1 = nn.Conv2d(in_feature,out_feature,kernel_size=1,stride=1, bias=False)
        ### 2nd ###
        self.conv2 = nn.Conv2d(out_feature,out_feature,kernel_size=kernel_size,stride=stride, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_feature)
        self.actv2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=kernel_size,stride=stride)

        self.covr1.weight.requires_grad = False

    def forward(self,X):
        R = X
        ### 1st ###
        X = self.actv1(self.norm1(self.conv1(X)))
        R = self.covr1(self.pool1(R))
        ### 2nd ###
        X = self.norm2(self.conv2(X))
        R = self.pool2(R)
        # print(X.shape,R.shape)
        X += R
        X = self.actv2(X)
        return X

class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()
        self.norm = nn.BatchNorm2d(1)
        self.block1 = Residual_Block(in_feature=  1, out_feature=  64, kernel_size = (5,5), stride = (1,3))
        self.block2 = Residual_Block(in_feature= 64, out_feature= 128, kernel_size = (4,4), stride = (1,2))
        self.block3 = Residual_Block(in_feature=128, out_feature= 256, kernel_size = (3,3), stride = (2,2))
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.norm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(self.pool(x), 1)
        return x

class AlexNet(nn.Module):
    
    def __init__(self, num_classes=8):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,
                                96,
                                kernel_size=11,
                                stride=4,
                                padding=4)),
            ('act1', nn.ReLU()),
            ('bn1', nn.BatchNorm2d(96)),
            ('pool1', nn.MaxPool2d(3, stride=2)),
            ('conv2', nn.Conv2d(96,
                                256,
                                kernel_size=5,
                                stride=1,
                                padding=2)),
            ('act2', nn.ReLU()),
            ('bn2', nn.BatchNorm2d(256)),
            ('pool2', nn.MaxPool2d(3, stride=2)),
            ('conv3', nn.Conv2d(256,
                                384, 
                                kernel_size=3,
                                stride=1,
                                padding=1)),
            ('act3', nn.ReLU()),
            ('bn3', nn.BatchNorm2d(384)),
            ('conv4', nn.Conv2d(384,
                                384,
                                kernel_size=3,
                                stride=1,
                                padding=1)),
            ('act4', nn.ReLU()),
            ('bn4', nn.BatchNorm2d(384)),
            ('conv5', nn.Conv2d(384,
                                256,
                                kernel_size=3,
                                stride=1,
                                padding=1)),
            ('act5', nn.ReLU()),
            ('bn5', nn.BatchNorm2d(256)),
            ('pool5', nn.MaxPool2d(3, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc6', nn.Linear(1024, 4096)),
            ('act6', nn.ReLU()),
            ('bn6', nn.BatchNorm1d(4096))
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(8192, 4096)),
            ('act7', nn.ReLU()),
            ('fc8', nn.Linear(4096, 4096)),
            ('act8', nn.ReLU()),
            ('fc9', nn.Linear(4096, num_classes))
        ]))
        
    def forward(self, input1, input2):
        patch1 = self.features(input1)
        patch2 = self.features(input2)
        patch = torch.cat([patch1, patch2], 1)
        out = self.classifier(patch)
        return out    
    
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, identity_downsample=None, stride=1):
        assert num_layers in [18, 34, 50, 101, 152], "should be a a valid architecture"
        super(Block, self).__init__()
        self.num_layers = num_layers
        if self.num_layers > 34:
            self.expansion = 4
        else:
            self.expansion = 1
        # ResNet50, 101, and 152 include additional layer of 1x1 kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if self.num_layers > 34:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        else:
            # for ResNet18 and 34, connect input directly to (3x3) kernel (skip first (1x1))
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        if self.num_layers > 34:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_layers, block, image_channels, num_classes):
        assert num_layers in [18, 34, 50, 101, 152], f'ResNet{num_layers}: Unknown architecture! Number of layers has ' \
                                                     f'to be 18, 34, 50, 101, or 152 '
        super(ResNet, self).__init__()
        if num_layers < 50:
            self.expansion = 1
        else:
            self.expansion = 4
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34 or num_layers == 50:
            layers = [3, 4, 6, 3]
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(num_layers, block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(num_layers, block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(num_layers, block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(num_layers, block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_layers, block, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(block(num_layers, self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion # 256
        for i in range(num_residual_blocks - 1):
            layers.append(block(num_layers, self.in_channels, intermediate_channels)) # 256 -> 64, 64*4 (256) again
        return nn.Sequential(*layers)
    
    
    