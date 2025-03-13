import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import *
from utils import load_data
from cnet import CNet
import glob
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from config import *
from augment import *
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter

def get_weighted_sampler(label_list, num_classes):
    counter = Counter(label_list)
    class_sample_count = np.array([counter.get(i, 0) for i in range(num_classes)])
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[l] for l in label_list])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler
    
if __name__ == '__main__':
    args = set_default_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = [i for i in range(0, len(gpu_id.split(",")))]
    print("Use {} GPU".format(len(gpu_id.split(","))))

    record_path = "record/"
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    
    model_name = f"ME{args.model_type}_{args.dataset}_{timestamp}"

    if args.suffix != None:
        model_name += f".{args.suffix}"
        

    if not os.path.exists(os.path.join(args.project_path, record_path, model_name, "model")):
        os.makedirs(os.path.join(args.project_path, record_path, model_name, "model"))

    if args.dataset == "ISIC":
        args.num_classes = 8
    elif args.dataset == "PathMNIST":
        args.num_classes = 9

    full_log_path = os.path.join(args.project_path, record_path, model_name, "log.log")
    log_file = open(full_log_path, "w+")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.writelines("============ Config =============\n")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
        log_file.writelines(f"{k}: {v}\n")
    log_file.writelines("=================================\n")
    log_file.close()
    print("============== Load Dataset ===============")
    
    if args.dataset == "ISIC":
        train_data, train_label = load_data(args.root_path, args.dataset, "train")
        print("data length", len(train_data))

        val_data, val_label = load_data(args.root_path, args.dataset, "test")
        print("data length", len(val_data))

    print("============== Model Setup ===============")

    train_transform = ISIC2019_Augmentations(is_training=True, image_size=args.img_size, input_size=args.img_size, model_name=model_name).transforms
    test_transform = ISIC2019_Augmentations(is_training=False, image_size=args.img_size, input_size=args.img_size, model_name=model_name).transforms
    if args.dataset == "PathMNIST":
        train_dataset = PathMNISTDataset(split="train", download=False, size=args.img_size, transform=train_transform)
        val_dataset = PathMNISTDataset(split="test", download=False, size=args.img_size, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                    pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=False)
    else:
        sampler = get_weighted_sampler(train_label, args.num_classes)
        train_dataset = ImageDataset(train_data, train_label, train_transform)
        val_dataset = ImageDataset(val_data, val_label, test_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=16,
                                    pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                                pin_memory=True, drop_last=False)
    
    net = CNet(args, train_loader, val_loader, full_log_path, record_path, model_name, gpuid)

    print("============== Start Training ===============")
    
    net.run()