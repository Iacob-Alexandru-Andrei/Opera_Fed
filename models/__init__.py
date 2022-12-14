import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
import torchvision
from models.self_supervised import *
from models.utils import *

def freeze_network(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    return model

#model = add_classifier(encoder,in_size=outsize,out_size=len(lb.classes_),freeze=freeze)    

def add_classifier(enc,in_size,out_size,freeze):
    if freeze == True:
        enc = freeze_network(enc)
    clf = Classifier(in_size,128,out_size)
    model = ED_module(encoder=enc,decoder=clf)
    return model

def add_SimCLR(enc,out_size):
    """
    enc: encoder(nn.Module)
    out_size: output size of encoder
    """
    clf = Projection_head(out_size,128,head='linear') # 'linear'
    model = SimCLR(enc,clf)
    return model


#############################################################

def add_SimCLR_multi(enc1, enc2, out_size1, out_size2):

    dec1 = Projection_head(out_size1, 128, head = 'linear') # 'linear'
    dec2 = Projection_head(out_size2, 128, head = 'linear')
    model = SimCLR_multi(enc1, enc2, dec1, dec2)
    return model

# def create_baseline_model():
#     out = 96
#     enc = Encoder([32,64,out])
#     model = add_classifier(enc,out_size=10*out,freeze=False)
#     return model

def create_encoder(network,pairing):
    if network == "shallow":
        if pairing == 'csi':
            encoder = create_baseline_encoder(scale_factor=1)
            outsize = 384 #960 #1152 # 2688
        elif pairing == 'nuc2':
            encoder = create_baseline_encoder(scale_factor=1)
            outsize = 384 #960 #1152 #2688
        elif pairing == 'pwr' or pairing == 'pwr1' or pairing == 'pwr2' or pairing == 'pwr3'or pairing == 'PWRPWR':
            encoder = create_baseline_encoder(scale_factor=1) # scale_factor=3
            outsize = 384 #1152
        else: 
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "alexnet":
        if pairing == 'csi':
            encoder,outsize = create_alexnet((1,4),scale_factor=1)
        elif pairing == 'nuc2':
            encoder,outsize = create_alexnet((1,4),scale_factor=1)
        elif pairing == 'pwr'or pairing == 'pwr1' or pairing == 'pwr2' or pairing == 'pwr3'or pairing == 'PWRPWR':
            encoder,outsize = create_alexnet((1,4),scale_factor=1)   # create_alexnet((4,1),scale_factor=2)
        else: 
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "resnet":
        if pairing == 'csi':
            encoder,outsize = create_resnet18((2,2))
        if pairing == 'nuc2':
            encoder,outsize = create_resnet18((2,2))
        elif pairing == 'pwr' or pairing == 'pwr1' or pairing == 'pwr2' or pairing == 'pwr3'or pairing == 'PWRPWR':
            encoder,outsize = create_resnet18((2,2))
        else: 
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
    elif network == "vgg16":
        if pairing == 'csi':
            encoder,outsize = create_vgg16((2,2))
        if pairing == 'nuc2':
            encoder,outsize = create_vgg16((2,2))
        elif pairing == 'pwr'or pairing == 'pwr1' or pairing == 'pwr2' or pairing == 'pwr3'or pairing == 'PWRPWR':
            encoder,outsize = create_vgg16((2,2))
        else: 
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")
                        
    elif network == "convnet":
        if pairing == 'csi':
            encoder = create_convnet(scale_factor=1)
            outsize = 4096
        if pairing == 'nuc2':
            encoder = create_convnet(scale_factor=1)
            outsize = 4096 
        elif pairing == 'pwr'or pairing == 'pwr1' or pairing == 'pwr2' or pairing == 'pwr3'or pairing == 'PWRPWR':
            encoder = create_convnet(scale_factor=1)
            outsize = 4096
        else: 
            raise ValueError("pairing must be in {'csi','nuc2','pwr'}")       
            
           
    else:
        raise ValueError("network must be in {'shallow','alexnet','resnet'}")
    return encoder, outsize
