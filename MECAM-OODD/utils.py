import os
import glob
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_data(dataset_root, dataset, split):
    if dataset == "ISIC":
        dataset_path = os.path.join(dataset_root, "ISIC", split)
        MEL = glob.glob(os.path.join(dataset_path, "MEL", "*.jpg"))
        NV = glob.glob(os.path.join(dataset_path, "NV", "*.jpg"))
        BCC = glob.glob(os.path.join(dataset_path, "BCC", "*.jpg"))
        AK = glob.glob(os.path.join(dataset_path, "AK", "*.jpg"))
        BKL = glob.glob(os.path.join(dataset_path, "BKL", "*.jpg"))
        DF = glob.glob(os.path.join(dataset_path, "DF", "*.jpg"))
        VASC = glob.glob(os.path.join(dataset_path, "VASC", "*.jpg"))
        SCC = glob.glob(os.path.join(dataset_path, "SCC", "*.jpg"))

        MEL.sort()
        NV.sort()
        BCC.sort()
        AK.sort()
        BKL.sort()
        DF.sort()
        VASC.sort()
        SCC.sort()

        label = [0]*len(MEL)+[1]*len(NV)+[2]*len(BCC)+[3]*len(AK)+[4]*len(BKL)+[5]*len(DF)+[6]*len(VASC)+[7]*len(SCC)
        print("Num of MEL Data:", len(MEL))
        print("Num of NV Data:", len(NV))
        print("Num of BCC Data:", len(BCC))
        print("Num of AK Data:", len(AK))
        print("Num of BKL Data:", len(BKL))
        print("Num of DF Data:", len(DF))
        print("Num of VASC Data:", len(VASC))
        print("Num of SCC Data:", len(SCC))
        data = MEL + NV + BCC + AK + BKL + DF + VASC + SCC
    elif dataset == "RSNA":
        dataset_path = os.path.join(dataset_root, "RSNA_Pneumonia", split)
        normal = glob.glob(os.path.join(dataset_path, "Normal", "*.png"))
        abnormal = glob.glob(os.path.join(dataset_path, "Abnormal", "*.png"))
        other = glob.glob(os.path.join(dataset_path, "Other", "*.png"))
        normal.sort()
        abnormal.sort()
        other.sort()
        label = [0]*len(normal)+[1]*len(abnormal)+[2]*len(other)
        print("Num of Abnormal Data:", len(abnormal))
        print("Num of Normal Data:", len(normal))
        print("Num of Other Data:", len(other))
        data = normal + abnormal + other
    elif dataset == "HeadCT":
        dataset_path = os.path.join(dataset_root, "HeadCT")
        normal = glob.glob(os.path.join(dataset_path, "normal", "*.png"))
        abnormal = glob.glob(os.path.join(dataset_path, "hemorrhage", "*.png"))
        normal.sort()
        abnormal.sort()
        label = [0]*len(normal)+[1]*len(abnormal)
        print("Num of Abnormal Data:", len(abnormal))
        print("Num of Normal Data:", len(normal))
        data = normal + abnormal
    elif dataset == "COVID-19":
        dataset_path = os.path.join(dataset_root, "COVID-19")
        normal = glob.glob(os.path.join(dataset_path, "Normal", "images", "*.png"))
        abnormal = glob.glob(os.path.join(dataset_path, "COVID", "images", "*.png"))
        normal.sort()
        abnormal.sort()
        label = [0]*len(normal)+[1]*len(abnormal)
        print("Num of Abnormal Data:", len(abnormal))
        print("Num of Normal Data:", len(normal))
        data = normal + abnormal
    return data, label

def plot_roc_curve(ind_conf, ood_conf, save_path):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate(
        (np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    fpr, tpr, _ = roc_curve(ind_indicator, conf)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve", color='blue', lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_score_distribution(ind_conf, ood_conf, save_path):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ind_conf, label="ID", color="blue", fill=True, alpha=0.5)
    sns.kdeplot(ood_conf, label="OOD", color="red", fill=True, alpha=0.5)

    # Labels and legend
    plt.title("Distribution of ID and OOD Scores", fontsize=16)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)

    # Show plot
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
