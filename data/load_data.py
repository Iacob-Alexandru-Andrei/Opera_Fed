#########
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import shutup

shutup.please()

#########
import numpy as np
import pandas as pd
import os
import re
import time

#########
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from sklearn.model_selection import train_test_split

#########
import matplotlib.pyplot as plt
from librosa.display import specshow


def concatenate_samples(filepaths, data_type="spectrogram"):
    mats = []
    if data_type == "spectrogram":
        for p in filepaths:
            arr = pd.read_csv(p, header=None).to_numpy()  # size (512, 512)
            arr = MinMaxScaler().fit_transform(arr)  # normalization
            arr = resize(
                arr, (224, 224), order=1, preserve_range=True
            )  # size (224, 224)
            arr = arr.reshape(-1, *arr.shape)  # size (1, 244, 244)
            mats.append(arr)
        return np.concatenate(mats, axis=0)
    else:
        for p in filepaths:
            arr = csv_to_arr(p)
            mats.append(arr)
        return np.array(mats)


def remove_str_nuc(filenames):
    rstr = r"_NUC1"
    return [[re.sub(rstr, "", filename)] for filename in filenames]


def remove_str_pwr(filenames):
    rstr = r"_" + "PWR_ch1"
    return [[re.sub(rstr, "", filename)] for filename in filenames]


def remove_str_kinect(filenames):
    rstr = r"_" + "Kinect_1"
    return [[re.sub(rstr, "", filename)] for filename in filenames]


############################################ 2 receivers / 2 features / 2 modalities ######################################################


def import_nuc_data(
    directory, data_directory, partition_str, data_type="spectrogram"
):  #########

    """
    import all data (in pair) in the directory
    """

    print("Importing Data ", end="")

    data = {"X1": [], "X2": [], "y": []}

    pwr_dic = f"{data_directory}/exp_15_pwr_spectrograms/"

    pwr_dic = pwr_dic + os.listdir(pwr_dic)[0]
    label_ls = os.listdir(pwr_dic)

    label_ls = os.listdir(directory)

    if ".ipynb_checkpoints" in label_ls:
        label_ls.remove(".ipynb_checkpoints")

    for label in label_ls:
        print(">", end="")

        # selcting available pairs
        pfiles_1 = filter_files(
            [
                f.split(".")[0]
                for f in os.listdir(pwr_dic + "/" + label + "/" + "pwr_ch1")
            ],
            partition_str,
        )  # get 1st modality file names
        pfiles_2 = filter_files(
            [
                f.split(".")[0]
                for f in os.listdir(directory + "/" + label + "/" + "nuc1")
            ],
            partition_str,
        )  # get 2nd modality file names
        print(pfiles_1, pfiles_2)
        pfiles_1_new = remove_str_pwr(pfiles_1)  # removes str characters
        pfiles_2_new = remove_str_nuc(pfiles_2)  # removes str characters

        available_pairs = np.intersect1d(pfiles_1_new, pfiles_2_new).tolist()

        # get csv names of the two sets
        files_1 = [
            directory
            + "/"
            + label
            + "/"
            + "nuc1"
            + "/"
            + pfilename
            + "_"
            + "NUC1"
            + ".csv"
            for pfilename in available_pairs
        ]
        files_2 = [
            directory
            + "/"
            + label
            + "/"
            + "nuc2"
            + "/"
            + pfilename
            + "_"
            + "NUC2"
            + ".csv"
            for pfilename in available_pairs
        ]

        # importing spectrograms/time-series
        X1 = concatenate_samples(files_1, data_type)
        X2 = concatenate_samples(files_2, data_type)

        # get labels
        y = np.full(X1.shape[0], label)

        assert X1.shape[0] == X2.shape[0]

        data["X1"].append(X1)
        data["X2"].append(X2)
        data["y"].append(y)

    print(" Complete")

    if data_type == "spectrogram":
        return (
            np.concatenate(data["X1"]).reshape(-1, 1, 224, 224),
            np.concatenate(data["X2"]).reshape(-1, 1, 224, 224),
            np.concatenate(data["y"]),
        )

    else:
        return (
            np.concatenate(data["X1"]),
            np.concatenate(data["X2"]),
            np.concatenate(data["y"]),
        )


################################################################################################################################


def filter_files(files, partition_str):
    return [file for file in files if partition_str in file]


def import_pwr_data(directory, data_directory, partition_str, data_type="spectrogram"):

    """
    import all data (in pair) in the directory
    """

    print("Importing Data ", end="")

    data = {"X1": [], "X2": [], "X3": [], "y": []}

    nuc_dic = f"{data_directory}/exp_10_amp_spec_only_STFT/"
    nuc_dic = nuc_dic + os.listdir(nuc_dic)[0]
    label_ls = os.listdir(nuc_dic)
    print(label_ls)

    if ".ipynb_checkpoints" in label_ls:
        label_ls.remove(".ipynb_checkpoints")

    for label in label_ls:

        print(">", end="")

        # selecting available pairs
        pfiles_1 = filter_files(
            [f.split(".")[0] for f in os.listdir(nuc_dic + "/" + label + "/" + "nuc1")],
            partition_str,
        )

        # get 1st modality file names
        pfiles_2 = filter_files(
            [
                f.split(".")[0]
                for f in os.listdir(directory + "/" + label + "/" + "pwr_ch1")
            ],
            partition_str,
        )
        # get 2nd modality file names
        print(pfiles_1, pfiles_2)

        pfiles_1_new = remove_str_nuc(pfiles_1)  # removes str characters
        pfiles_2_new = remove_str_pwr(pfiles_2)  # removes str characters

        available_pairs = np.intersect1d(pfiles_1_new, pfiles_2_new).tolist()

        # get csv names of the two sets
        files1 = [
            directory
            + "/"
            + label
            + "/"
            + "pwr_ch1"
            + "/"
            + pfilename
            + "_"
            + "PWR_ch1"
            + ".csv"
            for pfilename in available_pairs
        ]
        files2 = [
            directory
            + "/"
            + label
            + "/"
            + "pwr_ch2"
            + "/"
            + pfilename
            + "_"
            + "PWR_ch2"
            + ".csv"
            for pfilename in available_pairs
        ]
        files3 = [
            directory
            + "/"
            + label
            + "/"
            + "pwr_ch3"
            + "/"
            + pfilename
            + "_"
            + "PWR_ch3"
            + ".csv"
            for pfilename in available_pairs
        ]

        # importing spectrograms/time-series
        X1 = concatenate_samples(files1, data_type)
        X2 = concatenate_samples(files2, data_type)
        X3 = concatenate_samples(files3, data_type)

        # get labels
        y = np.full(X1.shape[0], label)

        data["X1"].append(X1)
        data["X2"].append(X2)
        data["X3"].append(X3)
        data["y"].append(y)

    print(" Complete")

    return (
        np.concatenate(data["X1"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["X2"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["X3"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["y"]),
    )


def import_kinect_data(
    directory, data_directory, partition_str, data_type="spectrogram"
):

    """
    import all data (in pair) in the directory
    """

    print("Importing Data ", end="")

    data = {"X1": [], "X2": [], "X3": [], "y": []}

    pwr_dic = f"{data_directory}/exp_15_pwr_spectrograms/"
    nuc_dic = f"{data_directory}/exp_10_amp_spec_only_STFT/"
    nuc_dic = nuc_dic + os.listdir(nuc_dic)[0]
    pwr_dic = pwr_dic + os.listdir(pwr_dic)[0]
    label_ls = os.listdir(nuc_dic)

    if ".ipynb_checkpoints" in label_ls:
        label_ls.remove(".ipynb_checkpoints")
    for label in label_ls:

        print(">", end="")

        # selecting available pairs
        pfiles_1 = filter(
            [f.split(".")[0] for f in os.listdir(nuc_dic + "/" + label + "/" + "nuc1")],
            partition_str,
        )  # get 1st modality file names
        pfiles_2 = filter(
            [
                f.split(".")[0]
                for f in os.listdir(directory + "/" + label + "/" + "ch1")
            ],
            partition_str,
        )  # get 2nd modality file names
        pfiles_3 = filter(
            [
                f.split(".")[0]
                for f in os.listdir(pwr_dic + "/" + label + "/" + "pwr_ch1")
            ],
            partition_str,
        )  # get 1st modality file names
        print(pfiles_1, pfiles_2, pfiles_3)
        pfiles_1_new = remove_str_nuc(pfiles_1)  # removes str characters
        pfiles_2_new = remove_str_kinect(pfiles_2)  # removes str characters
        pfiles_3_new = remove_str_pwr(pfiles_3)  # removes str characters

        available_pairs = np.intersect1d(pfiles_1_new, pfiles_2_new).tolist()

        # get csv names of the two sets
        files1 = [
            directory
            + "/"
            + label
            + "/"
            + "ch1"
            + "/"
            + pfilename
            + "_"
            + "Kinect_1"
            + ".csv"
            for pfilename in available_pairs
        ]
        files2 = [
            directory
            + "/"
            + label
            + "/"
            + "ch2"
            + "/"
            + pfilename
            + "_"
            + "Kinect_1"
            + ".csv"
            for pfilename in available_pairs
        ]
        files3 = [
            directory
            + "/"
            + label
            + "/"
            + "ch3"
            + "/"
            + pfilename
            + "_"
            + "Kinect_1"
            + ".csv"
            for pfilename in available_pairs
        ]

        # importing spectrograms/time-series
        X1 = concatenate_samples(files1, data_type)
        X2 = concatenate_samples(files2, data_type)
        X3 = concatenate_samples(files3, data_type)

        # get labels
        y = np.full(X1.shape[0], label)

        data["X1"].append(X1)
        data["X2"].append(X2)
        data["X3"].append(X3)
        data["y"].append(y)

    print(" Complete")

    return (
        np.concatenate(data["X1"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["X2"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["X3"]).reshape(-1, 1, 224, 224),
        np.concatenate(data["y"]),
    )


def import_multiple_modalities(
    data_directory, partition_str, namings=[], data_type="spectrogram"
):

    """
    Import dataset of two receivers for multiple CSI modalities
    data_type: 'spectrogram'
    namings:[ 'exp_7_amp_spec_only', 'exp_9_phdiff_spec_only','exp_10_amp_spec_only_STFT', 'exp_11_phdiff_spec_only_STFT']
    """

    multimodal_data = {}
    i = 0

    for naming in namings:

        directory = f"{data_directory}/" + naming + "/"
        print(directory)
        if len(os.listdir(directory)) == 1:
            filepath = directory + os.listdir(directory)[0]
            if naming == "exp_15_pwr_spectrograms":
                datas = import_pwr_data(
                    filepath,
                    data_directory=data_directory,
                    data_type="spectrogram",
                    partition_str=partition_str,
                )
                multimodal_data["modal" + str(i)] = datas
                i = i + 1
                del datas
            elif naming == "exp_16_kinect_spectrograms":
                datas = import_kinect_data(
                    filepath,
                    data_directory=data_directory,
                    data_type="spectrogram",
                    partition_str=partition_str,
                )
                multimodal_data["modal" + str(i)] = datas
                i = i + 1
                del datas
            else:
                datas = import_nuc_data(
                    filepath,
                    data_directory=data_directory,
                    data_type="spectrogram",
                    partition_str=partition_str,
                )
                multimodal_data["modal" + str(i)] = datas
                i = i + 1
                del datas

        if len(os.listdir(directory)) == 2:
            filepath = directory + os.listdir(directory)[0]
            datas = import_nuc_data(
                filepath,
                data_directory=data_directory,
                data_type="spectrogram",
                partition_str=partition_str,
            )
            multimodal_data["modal" + str(i)] = datas
            del datas
            i = i + 1

            filepath = directory + os.listdir(directory)[1]
            datas = import_nuc_data(
                filepath,
                data_directory=data_directory,
                data_type="spectrogram",
                partition_str=partition_str,
            )
            multimodal_data["modal" + str(i)] = datas
            i = i + 1
            del datas

    return multimodal_data


def import_raw_csi_ts(
    directory,
    directory2,
    modal=["nuc1", "nuc2"],
    return_id=True,
    data_type="time-series",
):  #########

    """
    import all data (in pair) in the directory
    """

    print("Importing Data ", end="")

    data = {
        "X1": [],
        "X2": [],
        "y": [],
        "personid": [],
        "roomid": [],
        "expnum": [],
    }  ########  ,'expnum':[]

    label_ls = os.listdir(directory)

    if ".ipynb_checkpoints" in label_ls:
        label_ls.remove(".ipynb_checkpoints")

    for label in label_ls:
        print(">", end="")

        # selcting available pairs
        pfiles_2 = [
            f.split(".")[0] for f in os.listdir(directory2 + "/" + label)
        ]  # get 2nd modality file names

        if ("nuc" in modal[0]) & ("nuc" in modal[1]):
            pfiles_2_new, modal_2 = remove_str_nuc(
                pfiles_2, modal[1]
            )  # removes str characters

        # get csv names of the two sets
        files_2 = [
            directory2 + "/" + label + "/" + pfilename[0] + ".csv"
            for pfilename in pfiles_2_new
        ]
        # importing spectrograms/time-series
        X2 = concatenate_samples(files_2, data_type)

        # get labels
        y = np.full(X2.shape[0], label)

        data["X2"].append(X2)
        data["y"].append(y)
    print(" Complete")
    return np.concatenate(data["X2"]), np.concatenate(data["y"])


def split_multimodal_data(multimodal_data, split=0.8, views="dissociated", axis=3):

    if views == "associated":

        if len(multimodal_data["modal0"]) == 3:
            X, y = (
                np.concatenate(
                    (multimodal_data["modal0"][0], multimodal_data["modal0"][1]),
                    axis=axis,
                ),
                multimodal_data["modal0"][2],
            )
        if len(multimodal_data["modal0"]) == 4:
            X, y = (
                np.concatenate(
                    (
                        multimodal_data["modal0"][0],
                        multimodal_data["modal0"][1],
                        multimodal_data["modal0"][2],
                    ),
                    axis=axis,
                ),
                multimodal_data["modal0"][3],
            )

        for key in list(multimodal_data.keys())[1:]:
            if len(multimodal_data[key]) == 4:
                modal_X = np.concatenate(
                    (
                        multimodal_data[key][0],
                        multimodal_data[key][1],
                        multimodal_data[key][2],
                    ),
                    axis=axis,
                )
            else:
                modal_X = np.concatenate(
                    (multimodal_data[key][0], multimodal_data[key][1]), axis=axis
                )
            del multimodal_data[key]
            X = np.concatenate((X, modal_X), axis=axis)
            del modal_X

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split, random_state=42
        )
        return X_train, X_test, y_train, y_test

    if views == "dissociated":

        if len(multimodal_data["modal0"]) == 3:
            X1, X2, y = (
                multimodal_data["modal0"][0],
                multimodal_data["modal0"][1],
                multimodal_data["modal0"][2],
            )
        if len(multimodal_data["modal0"]) == 4:
            X, y = (
                np.concatenate(
                    (
                        multimodal_data["modal0"][0],
                        multimodal_data["modal0"][1],
                        multimodal_data["modal0"][2],
                    ),
                    axis=axis,
                ),
                multimodal_data["modal0"][3],
            )

        if len(multimodal_data["modal0"]) == 3:
            for key in list(multimodal_data.keys())[1:]:
                X1 = np.concatenate((X1, multimodal_data[key][0]), axis=axis)
                X2 = np.concatenate((X2, multimodal_data[key][1]), axis=axis)

        if len(multimodal_data["modal0"]) == 4:
            for key in list(multimodal_data.keys())[1:]:
                X1 = np.concatenate((X, multimodal_data[key][0]), axis=axis)
                X2 = np.concatenate((X, multimodal_data[key][1]), axis=axis)

        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
            X1, X2, y, train_size=split, random_state=42
        )
        return X1_train, X1_test, X2_train, X2_test, y_train, y_test
