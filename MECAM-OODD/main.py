import argparse
import os, glob
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from utils import load_data
from dataset import *
from engine import Engine
import csv
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path",
                        type=str, 
                        default="../../",
                        help="path of project")
                        
    parser.add_argument("--root_path",
                        type=str, 
                        default="/home/vincent/Dataset/",
                        help="path of dataset")

    parser.add_argument("--id_dataset",
                        "-id",
                        type=str,
                        choices=["ISIC", "PathMNIST"], 
                        default="ISIC",
                        help="id dataset")  

    parser.add_argument("--ood_dataset",
                        "-ood",
                        type=str,
                        choices=["HeadCT", "COVID-19", "iSUN", "RSNA"], 
                        default="HeadCT",
                        help="ood dataset")   

    parser.add_argument("--selected_exit",
                        type=int,
                        default=[0,1,2,3],
                        nargs='+',
                        help="selected exit")  

    parser.add_argument("--img_size",
                        type=int,
                        default=224,
                        help="image size")

    parser.add_argument("--pretrained_path",
                        type=str, 
                        default="",
                        help="pretrained path")

    parser.add_argument("--gpu_id",
                        "-gid",
                        type=str,
                        default ='',
                        help="gpu id number")   

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")


    # args parse
    args = parser.parse_args()
    gpu_id = args.gpu_id
    if gpu_id == '':
        gpu_id = ",".join([str(g) for g in np.arange(torch.cuda.device_count())])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpuid = [i for i in range(0, len(gpu_id.split(",")))]
    print("Use {} GPU".format(len(gpu_id.split(","))))

    record_path = f"record/{args.id_dataset}"

    if args.id_dataset == "ISIC":
        args.num_classes = 8
    elif args.id_dataset == "PathMNIST":
        args.num_classes = 9
        
    print("============== Load Dataset ===============")
    
    if args.id_dataset == "ISIC":
        id_data, id_label = load_data(args.root_path, args.id_dataset, "test")
        print("data length", len(id_data))

    if args.ood_dataset == "iSUN":
        dataset_path = os.path.join("../../iSUN_data", "iSUN_patches")
        ood_data = glob.glob(os.path.join(dataset_path, "*.jpeg"))
        ood_label = [0]*len(ood_data)
        print("data length", len(ood_data))
    else:
        ood_data, ood_label = load_data(args.root_path, args.ood_dataset, "test")
        print("data length", len(ood_data))

    print("============== Start Testing ===============")

    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    ])
    if args.id_dataset == "PathMNIST":
        id_dataset = PathMNISTDataset(split="test", download=False, size=args.img_size, transform=test_transform)
    else:
        id_dataset = ImageDataset(id_data, id_label, test_transform)

    id_loader = DataLoader(id_dataset, batch_size=128, shuffle=False, num_workers=16,
                        pin_memory=True, drop_last=False)

    ood_dataset = ImageDataset(ood_data, ood_label, test_transform)

    ood_loader = DataLoader(ood_dataset, batch_size=128, shuffle=False, num_workers=16,
                            pin_memory=True, drop_last=False)

    engine = Engine(args, id_loader, ood_loader, record_path, gpuid)
    