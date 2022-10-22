import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision
#from models.utils import *

class DataAugmentation(nn.Module):

    def __init__(self, size):
        super(DataAugmentation, self).__init__()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=size),
#             torchvision.transforms.ToTensor()
        ])

    def forward(self, x):
        return self.transform(x), self.transform(x)


class Projection_head(nn.Module):
    """
    Projection head:

    head: linear/mlp
    """
    def __init__(self,dim_in,feat_dim,head='mlp'): # 'linear'
        super(Projection_head, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self,X):
        X = self.head(X)
        X = F.normalize(X, dim=1)
        return X

#####################   SIM - SIAM   #######################    
class ProjectionMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class PredictorMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)    
    
    
class SimSiam(nn.Module):

    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder, self.outsize = create_encoder(backbone,'nuc2') # Encoder(backbone=backbone, pretrained=False)

        # Projection (mlp) network
        self.projection_mlp =   ProjectionMLP(
            input_dim       =   self.outsize,     #   self.encoder.emb_dim,
            hidden_dim      =   proj_hidden_dim,
            output_dim      =   latent_dim
        )

        # Predictor network (h)
        self.predictor_mlp = PredictorMLP(
            input_dim=latent_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=latent_dim
        )

    def forward(self, x: torch.Tensor):
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)   
    
    
##############################################################    
class SimCLR(nn.Module):
    def __init__(self, encoder, decoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,t): # tuple
        batch_size = t[0].shape[0]
        t = torch.cat(t,dim=0) # tensor
        t = self.encoder(t)
        t = self.decoder(t)
        t = torch.split(t,batch_size,dim=0)
        return t # tuple
    

class SimCLR_multi(nn.Module):
    def __init__(self, enc1, enc2, dec1, dec2):
        super(SimCLR_multi, self).__init__()
        self.encoder = enc1
        self.decoder = dec1
        self.encoder2 = enc2
        self.decoder2 = dec2

    def forward(self, t): # tuple
        o1 = self.encoder(t[0])
        o1 = self.decoder(o1)
        o2 = self.encoder2(t[1])
        o2 = self.decoder2(o2)
        return o1, o2 # tuple
    
##############################################################    
import torch.nn as nn
import torchvision


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR_2(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self): # , args
        super(SimCLR_2, self).__init__()

        #self.args = args

        self.encoder = self.get_resnet("resnet18")

        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.encoder.fc = Identity()  # remove fully-connected layer after pooling layer

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, 128, bias=False), #args.projection_dim
        )
        

    def get_resnet(self, name):
        resnets = {
            "resnet18": torchvision.models.resnet18(),
            "resnet50": torchvision.models.resnet50(),
        }
        if name not in resnets.keys():
            raise KeyError(f"{name} is not a valid ResNet version")
        return resnets[name]


    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)

        if self.args.normalize:
            z = nn.functional.normalize(z, dim=1)
        return h, z    