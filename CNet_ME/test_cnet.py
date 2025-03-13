import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils import load_data
from dataset import *
from cnet import CNet
import glob
from datetime import datetime
from config import *
from augment import *

if __name__ == '__main__':
    args = set_default_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = [i for i in range(0, len(gpu_id.split(",")))]
    print("Use {} GPU".format(len(gpu_id.split(","))))

    record_path = "record/"

    model_name = args.pretrained_path
    assert os.path.exists(os.path.join(args.project_path, record_path, model_name, "model")), f'{model_name} model not found'

    if args.dataset == "ISIC":
        args.num_classes = 8
    elif args.dataset == "PathMNIST":
        args.num_classes = 9


    for k, v in vars(args).items():
        print(f"{k}: {v}")
        
    full_log_path = os.path.join(args.project_path, record_path, model_name, "log.log")
    print("============== Load Dataset ===============")
    
    if args.dataset == "ISIC":
        train_data, train_label = load_data(args.root_path, args.dataset, "train")
        print("data length", len(train_data))

        val_data, val_label = load_data(args.root_path, args.dataset, "test")
        print("data length", len(val_data))

    print("============== Model Setup ===============")


    test_transform = ISIC2019_Augmentations(is_training=False, image_size=args.img_size, input_size=args.img_size, model_name=model_name).transforms
    if "MNIST" in args.dataset:
        if args.dataset == "PathMNIST":
            evaluate_dataset = PathMNISTDataset(split="test", download=False, size=args.img_size, transform=test_transform)
        elif args.dataset == "OrganCMNIST":
            evaluate_dataset = OrganCMNISTDataset(split="test", download=False, size=args.img_size, transform=test_transform)
    else:        
        evaluate_dataset = ImageDataset(val_data, val_label, test_transform)

    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                        pin_memory=True, drop_last=False)

    

    net = CNet(args, None, None, None, record_path, model_name, gpuid)
    print("============== Start Testing ===============")
    
    test_loss, test_auc = net.test(evaluate_loader)
    log_file = open(full_log_path, "a")
    log_file.writelines(
        f"Final !! Test Loss: {test_loss}, Test AUC: {test_auc:.4f}\n")
    log_file.writelines(str(datetime.now())+"\n")
    log_file.close()

    print("============== Start Get Feature ===============")
    test_transform = ISIC2019_Augmentations(is_training=False, image_size=args.img_size, input_size=args.img_size, model_name=model_name).transforms
    if args.dataset == "PathMNIST":
        evaluate_dataset = PathMNISTDataset(split="test", download=False, size=args.img_size, transform=test_transform)
    else:        
        evaluate_dataset = ImageDataset(val_data, val_label, test_transform)

    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16,
                        pin_memory=True, drop_last=False)
    net.output_id_feature(evaluate_loader, model_name)