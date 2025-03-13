import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

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
        normal.sort()
        abnormal.sort()
        label = [0]*len(normal)+[1]*len(abnormal)
        print("Num of Abnormal Data:", len(abnormal))
        print("Num of Normal Data:", len(normal))
        data = normal + abnormal
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


def save_chart(epochs, train_list, val_list, save_path, name=''):
    x = np.arange(epochs)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    lns1 = ax.plot(x, train_list, 'b', label='train {}'.format(name))
    lns2 = ax.plot(x, val_list, 'r', label='val {}'.format(name))
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]

    ax.legend(lns, labs, loc='upper right')
    ax.set_xlabel("Epochs")
    ax.set_ylabel(name)

    plt.savefig(save_path)
    plt.close()

def output_historgram(pred_class, save_path, name=''):
    data = pred_class.numpy()
    unique_values, counts = np.unique(data, return_counts=True)
    fig = plt.figure(figsize=(7, 5))
    # Plot the histogram
    plt.bar(unique_values, counts, color='skyblue', edgecolor='black')

    plt.xticks(unique_values)

    # Add labels and title
    plt.xlabel('Values (Predicted Class)')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted Classes')

    # Display the plot
    plt.savefig(save_path)
    plt.close()