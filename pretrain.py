import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F
import time
from tqdm import tqdm
from data import prepare_single_source
from dino_utils import clip_gradients
from dino_evaluation import compute_embedding, compute_knn,compute_embedding
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import pathlib
from visualize_tb import compute_embedding
source_dir = "/nfs-share/aai30/projects/transformer_baseline"
##################################################################################################

def pretrain(model, optimizer, lr_scheduler, epochs, train_loader, valid_loader, device, exp_name, lb, embedding = 'save'):

    t = time.time()

    record = {'train_loss':[], 'val_loss':[]}
    
    tensorboard_dir = f"{source_dir}/logs/pretrain/" + exp_name
    
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    
    logging_path = pathlib.Path(tensorboard_dir)
    tb = SummaryWriter(logging_path)

    ####################################

    for i, epoch in enumerate(range(epochs)):
        epoch_loss = 0
        print(f"Epoch {i+1}: \n0% ", end = '')
        for idx, data in enumerate(train_loader):
            print('>',end='')
            data  = data.to(device).float()
            loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_scheduler.step_update(epoch * len(train_loader) + idx)
            
            epoch_loss += loss / len(train_loader)

            if device:
                data = data.cpu()
                loss = loss.cpu()
                del data, loss

        epoch_loss = epoch_loss.tolist()      

        record['train_loss'].append(epoch_loss)        
        
        with torch.no_grad():
            epoch_val_loss = 0
            
            for data, _ in valid_loader:
                data = data.float().to(device)

                val_loss   = model(data)
                epoch_val_loss += val_loss / len(valid_loader)

                if device:
                    data = data.cpu()
                    val_loss = val_loss.cpu()
                    del data, val_loss
                    
            epoch_val_loss  = epoch_val_loss.tolist()    

            record['val_loss'].append(epoch_val_loss)
            
            if tb != None:
                os.chdir(tensorboard_dir)
                tb.add_scalar("Train Loss", epoch_loss, epoch)
                tb.add_scalar("Val Loss", epoch_val_loss, epoch)
                
                if embedding == 'save':
                    embs, imgs, labels_ = compute_embedding(model, valid_loader, lb)

                    tb.add_embedding(
                        embs,
                        metadata = labels_,
                        label_img = imgs,
                        global_step = epoch,
                        tag = "embeddings")

        if min(record['val_loss']) == record['val_loss'][i]:
            save_model(model.encoder, exp_name = exp_name, filename = record['val_loss'][i])
            if embedding == 'save':
                os.chdir(f'{source_dir}')
        print(' 100%')

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f}\n")
        
    print(time.time() - t)
    
    del optimizer
    
    return model, record


def evaluation(model,test_loader,label_encoder=None):
    model = model.cpu()
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test, y_test
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
    cls = classification_report(y_test.view(-1), predicted.view(-1), output_dict=True)
    cls = pd.DataFrame(cls)
    print(cls)
    cmtx = confusion_matrix(y_test.view(-1), predicted.view(-1))
    cmtx = cmtx_table(cmtx,label_encoder)
    return cmtx,cls

def cmtx_table(cmtx,label_encoder=None):
    if label_encoder != None:
        cmtx = pd.DataFrame(cmtx,
                            index=[f"actual: {i}"for i in label_encoder.classes_.tolist()],
                            columns=[f"predict : {i}"for i in label_encoder.classes_.tolist()])
    else:
        cmtx = pd.DataFrame(cmtx)
    return cmtx

def save_model(model, exp_name, filename):
    os.chdir(f'{source_dir}/')
    outpath = 'results/saved_models/pretrain/' + exp_name
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        os.chdir(outpath + '/')
        torch.save(model.state_dict(), str(filename))
    else:
        os.chdir(outpath + '/')
        if filename < float(os.listdir()[0]):
            os.remove(os.listdir()[0])
            torch.save(model.state_dict(), str(filename))
    