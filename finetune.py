import os
import sys
from time import gmtime, strftime
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, nn
from torch.nn import functional as F
import time
from tqdm import tqdm
from data import prepare_single_source
from dino_utils import clip_gradients
from dino_evaluation import compute_embedding, compute_knn, compute_embedding
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import pathlib
from visualize_tb import compute_embedding

source_dir = "/nfs-share/aai30/projects/transformer_baseline"


def finetune(
    model,
    criterion,
    lr_scheduler,
    optimizer,
    epochs,
    train_loader,
    valid_loader,
    device,
    exp_name_ft,
    lb,
    embedding="save",
):

    t = time.time()

    record = {"train_loss": [], "train_f1": [], "val_loss": [], "f1_macro": []}

    tensorboard_dir = f"{source_dir}/logs/finetune/" + exp_name_ft

    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging_path = pathlib.Path(tensorboard_dir)
    tb = SummaryWriter(logging_path)

    ####################################

    for i, epoch in enumerate(range(epochs)):
        print(f"Epoch {i+1}: \n0% ", end="")
        epoch_loss = 0
        epoch_f1macro = 0

        for idx, (data, label) in enumerate(train_loader):
            print(">", end="")
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            lr_scheduler.step_update(epoch * len(train_loader) + idx)
            epoch_f1macro += f1_score(
                label.cpu().view(-1),
                torch.max(output.cpu(), 1)[1].view(-1),
                average="macro",
            ) / len(train_loader)
            epoch_loss += loss / len(train_loader)

            if device:
                data = data.cpu()
                loss = loss.cpu()
                output = output.cpu()
                label = label.cpu()
                del data, loss, label, output

        epoch_loss = epoch_loss.tolist()
        epoch_f1macro = epoch_f1macro.tolist()

        record["train_loss"].append(epoch_loss)
        record["train_f1"].append(epoch_f1macro)

        with torch.no_grad():
            epoch_val_loss = 0
            val_f1macro = 0

            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)
                epoch_val_loss += val_loss / len(valid_loader)

                predicted = torch.max(val_output.cpu(), 1)[1]
                val_f1macro += f1_score(
                    label.cpu().view(-1), predicted.view(-1), average="macro"
                ) / len(valid_loader)

                if device:
                    data = data.cpu()
                    val_loss = val_loss.cpu()
                    val_output = val_output.cpu()
                    label = label.cpu()
                    del data, val_loss, label, val_output

            val_f1macro = val_f1macro.tolist()
            epoch_val_loss = epoch_val_loss.tolist()

            record["f1_macro"].append(val_f1macro)
            record["val_loss"].append(epoch_val_loss)

            if tb != None:
                os.chdir(tensorboard_dir)
                tb.add_scalar("Train Loss", epoch_loss, epoch)
                tb.add_scalar("Train Macro F1", epoch_f1macro, epoch)
                tb.add_scalar("Val Loss", epoch_val_loss, epoch)
                tb.add_scalar("Val Macro F1", val_f1macro, epoch)

                if embedding == "save":
                    embs, imgs, labels_ = compute_embedding(model, valid_loader, lb)

                    tb.add_embedding(
                        embs,
                        metadata=labels_,
                        label_img=imgs,
                        global_step=epoch,
                        tag="embeddings",
                    )

        if max(record["f1_macro"]) == record["f1_macro"][i]:
            save_model(model, exp_name_ft=exp_name_ft, filename=record["f1_macro"][i])
            if embedding == "save":
                os.chdir(f"{source_dir}")
        print(" 100%")
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - f1: {epoch_f1macro:.4f} - val_loss : {epoch_val_loss:.4f} - val_f1_macro: {val_f1macro:.4f}\n"
        )
    print(time.time() - t)

    del optimizer, criterion

    return model, record


def evaluation(model, test_loader, label_encoder=None):
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
    cmtx = cmtx_table(cmtx, label_encoder)
    return cmtx, cls


def cmtx_table(cmtx, label_encoder=None):
    if label_encoder != None:
        cmtx = pd.DataFrame(
            cmtx,
            index=[f"actual: {i}" for i in label_encoder.classes_.tolist()],
            columns=[f"predict : {i}" for i in label_encoder.classes_.tolist()],
        )
    else:
        cmtx = pd.DataFrame(cmtx)
    return cmtx


def save_model(model, exp_name_ft, filename):
    os.chdir(f"{source_dir}/")
    outpath = "results/saved_models/finetune/" + exp_name_ft
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        os.chdir(outpath + "/")
        torch.save(model.state_dict(), str(filename))
    else:
        os.chdir(outpath + "/")
        if filename > float(os.listdir()[0]):
            os.remove(os.listdir()[0])
            torch.save(model.state_dict(), str(filename))


def record_log(exp_name_ft, metrics, model_parameters, data, num_parameters):
    record_outpath = f"{source_dir}/results/records/finetune"
    prefix = record_outpath + "/" + exp_name_ft
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    metrics.to_csv(prefix + "/" + "metrics.csv")
    model_parameters.to_csv(prefix + "/" + "model_parameters.csv")
    data.to_csv(prefix + "/" + "data.csv")
    num_parameters.to_csv(prefix + "/" + "num_params.csv")
    print(f"Records saved in {exp_name_ft} folder")
