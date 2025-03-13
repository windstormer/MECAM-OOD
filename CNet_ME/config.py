import argparse


def set_default_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_path",
                        type=str, 
                        default="../../",
                        help="path of project")
    
    parser.add_argument("--root_path",
                        type=str, 
                        default="/home/vincent/Dataset/",
                        help="path of dataset")

    parser.add_argument("--dataset",
                        "-d",
                        type=str,
                        choices=["ISIC", "PathMNIST"], 
                        default="ISIC",
                        help="dataset")                    

    parser.add_argument("--gpu_id",
                        "-gid",
                        type=str,
                        default ='',
                        help="gpu id number")

    parser.add_argument("--epochs",
                        "-e",
                        type=int,
                        default=10,
                        help="number of epoch")

    parser.add_argument("--batch_size",
                        "-b",
                        type=int,
                        default=256,
                        help="batch size") 

    parser.add_argument("--learning_rate",
                        "-lr",
                        type=float,
                        default=0.01,
                        help="learning rate")   

    parser.add_argument("--img_size",
                        type=int,
                        default=224,
                        help="image size")   

    parser.add_argument("--wd",
                        type=float,
                        default=1e-4,
                        help="weight decay")

    parser.add_argument("--droprate",
                        type=float,
                        default=0.0,
                        help="dropout rate")            

    parser.add_argument("--suffix",
                        '-s',
                        type=str,
                        default=None,
                        help="suffix")

    parser.add_argument("--model_type",
                        type=str,
                        default="Res18",
                        choices=["Res18", "Res50"],
                        help="Type of Model")
    
    parser.add_argument("--pretrained_path",
                        type=str, 
                        default=None,
                        help="pretrained path")
    # args parse
    args = parser.parse_args()
    return args