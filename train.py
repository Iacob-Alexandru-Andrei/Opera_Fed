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

# from dino_utils import clip_gradients
# from dino_evaluation import compute_embedding, compute_knn,compute_embedding
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
from torch.nn import Module
from typing import Callable, Tuple

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import pathlib
from visualize_tb import compute_embedding

#####################################################################################################################


source_dir = "/nfs-share/aai30/projects/transformer_baseline"


def train_client(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    optimizer,
    criterion,
):
    """Train the network on the training set."""
    net.train()
    for _ in range(epochs):
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            loss.backward()
            optimizer.step()


def test_client(
    net: Module,
    testloader: DataLoader,
    device: str,
    criterion,
) -> Tuple[float, float]:
    """Validate the network on a test set."""
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            data, labels = data[0].to(device), data[1].to(device)
            outputs = net(data)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def train(
    model,
    criterion,
    optimizer,
    epochs,
    train_loader,
    valid_loader,
    device,
    exp_name,
    lb,
    embedding="save",
):

    t = time.time()

    record = {"train_loss": [], "train_f1": [], "val_loss": [], "f1_macro": []}

    tensorboard_dir = f"{source_dir}/logs/train/" + exp_name

    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    logging_path = pathlib.Path(tensorboard_dir)
    tb = SummaryWriter(logging_path)

    ####################################

    for i, epoch in enumerate(range(epochs)):
        epoch_loss = 0
        epoch_f1macro = 0

        for data, label in tqdm(train_loader):

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            save_model(model, exp_name=exp_name, filename=record["f1_macro"][i])
            if embedding == "save":
                os.chdir(f"{source_dir}")
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - f1: {epoch_f1macro:.4f} - val_loss : {epoch_val_loss:.4f} - val_f1_macro: {val_f1macro:.4f}\n"
        )
    print(time.time() - t)

    del optimizer, criterion

    return model, record


def short_evaluation(model, test_loader, device):
    # copy the model to cpu
    if device:
        model = model.cpu()
    acc = {"accuracy": 0, "f1_weighted": 0, "f1_macro": 0}
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            predicted = torch.max(y_val, 1)[1]
            acc["accuracy"] = accuracy_score(y_test.view(-1), predicted.view(-1))
            acc["f1_weighted"] = f1_score(
                y_test.view(-1), predicted.view(-1), average="weighted"
            )
            acc["f1_macro"] = f1_score(
                y_test.view(-1), predicted.view(-1), average="macro"
            )
    # send model back to gpu
    if device:
        model = model.to(device)
    return acc


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


def make_directory(name, epoch=None, filepath="./"):
    time = strftime("%Y_%m_%d_%H_%M", gmtime())
    directory = filepath + name + "_checkpoint_" + str(epoch) + "__" + time
    return directory


def save_checkpoint(model, optimizer, epoch, directory):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        directory,
    )
    print(f"save checkpoint in : {directory}")
    return


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer


# -----------------------------------------helper---------------------------------------


def record_log(
    record_outpath,
    exp_name,
    phase,
    record="None",
    cmtx="None",
    cls="None",
    loss_rec=True,
    acc_rec=False,
):
    prefix = record_outpath + "/" + exp_name + "_Phase_" + phase
    if type(record) != str:
        if loss_rec:
            pd.DataFrame(record["loss"], columns=["loss"]).to_csv(prefix + "_loss.csv")
        if acc_rec:
            df = pd.concat(
                (
                    pd.DataFrame(record["accuracy"]),
                    pd.DataFrame(record["f1_weighted"]),
                    pd.DataFrame(record["f1_macro"]),
                ),
                axis=1,
            )
            df.columns = ["accuracy", "f1_weighted", "f1_macro"]
            df.to_csv(prefix + "_accuracy.csv")
    if type(cmtx) != str:
        cmtx.to_csv(prefix + "_cmtx.csv")
    if type(cls) != str:
        cls.to_csv(prefix + "_report.csv")
    return


def save_model(model, exp_name, filename):
    os.chdir(f"{source_dir}/")
    outpath = "results/saved_models/train/" + exp_name
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        os.chdir(outpath + "/")
        torch.save(model.state_dict(), str(filename))
    else:
        os.chdir(outpath + "/")
        if filename > float(os.listdir()[0]):
            os.remove(os.listdir()[0])
            torch.save(model.state_dict(), str(filename))


#############################################################################################

# def train_dino_knn(
#     model,
#     train_loader,
#     test_loader,
#     criterion,
#     optimizer,
#     end,
#     start=1,
#     device=None,
#     regularize=None,
#     **kwargs,
# ):
#     """
#     training Loop
#     Arguments
#     model (torch.nn.Module): the model
#     train_loader (torch.utils.Dataset): torch.utils.Dataset of the training set
#     criterion (nn.Module): the loss function
#     optimizer (torch.optim): the optimizer to backpropagate the network
#     end (int): the epoch which the loop end after
#     start (int): the epoch which the loop start at
#     test_loader (torch.utils.Dataset): torch.utils.Dataset of the validation set
#     regularize (bool): add l2 regularization loss on top of the total loss
#     device (str): the device whic the model is allocated
#     Returns:
#     model (torch.nn.Module): trained model
#     record (dict): the record of the training, currently have
#     'loss' (every epoch), 'accuracy' (every 10 epochs),'f1_weighted' (every 10 epochs),'f1_macro (every 10 epochs)'
#     """

#     # Check device setting
#     if device:
#         model = model.to(device)  # full model with linear classifier

#     print("Start Training")
#     record = {"loss": [], "accuracy": [], "f1_weighted": [], "f1_macro": []}
#     i = start

#     # Loop
#     while i <= end:
#         print(f"Epoch {i}: ", end="")
#         for b, (X_train, y_train) in enumerate(train_loader):

#             if device:
#                 X_train = X_train.to(device)

#             print(f">", end="")

#             optimizer.zero_grad()
#             y_pred = model(X_train)

#             if device:
#                 X_train = X_train.cpu()
#                 del X_train
#                 y_train = y_train.to(device)

#             loss = criterion(y_pred, y_train)

#             if regularize:
#                 loss += reg_loss(model, device)

#             loss.backward()
#             optimizer.step()

#             if device:
#                 y_pred = y_pred.cpu()
#                 y_train = y_train.cpu()
#                 del y_pred, y_train

#         # One epoch completed
#         loss = loss.tolist()
#         record["loss"].append(loss)
#         print(f" loss: {loss} ", end="")

#         if (test_loader != None) and i % 10 == 0:
#             model.eval()
#             current_acc, current_f1_weighted, current_f1_macro = compute_knn_new(
#                 model.encoder,
#                 train_loader,
#                 test_loader,
#             )

#             record["accuracy"].append(current_acc)
#             record["f1_weighted"].append(current_f1_weighted)
#             record["f1_macro"].append(current_f1_macro)
#             display = current_acc
#             print(f" accuracy: {display}")

#             model.train()

#         i += 1

#     model = model.cpu()

#     return model, record
