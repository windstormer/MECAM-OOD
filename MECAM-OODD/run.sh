# #! /bin/bash
python3 main.py -id ISIC -ood iSUN --pretrained_path MERes18_ISIC_1231_003419 -gid 1
python3 main.py -id ISIC -ood iSUN --pretrained_path MERes50_ISIC_0114_155038 -gid 1
python3 main.py -id PathMNIST -ood iSUN --pretrained_path MERes18_PathMNIST_0113_015355 -gid 1
python3 main.py -id PathMNIST -ood iSUN --pretrained_path MERes50_PathMNIST_0113_104937 -gid 1

python3 main.py -id ISIC -ood HeadCT --pretrained_path MERes18_ISIC_1231_003419 -gid 1
python3 main.py -id ISIC -ood HeadCT --pretrained_path MERes50_ISIC_0114_155038 -gid 1
python3 main.py -id PathMNIST -ood HeadCT --pretrained_path MERes18_PathMNIST_0113_015355 -gid 1
python3 main.py -id PathMNIST -ood HeadCT --pretrained_path MERes50_PathMNIST_0113_104937 -gid 1

python3 main.py -id ISIC -ood COVID-19 --pretrained_path MERes18_ISIC_1231_003419 -gid 0
python3 main.py -id ISIC -ood COVID-19 --pretrained_path MERes50_ISIC_0114_155038 -gid 0
python3 main.py -id PathMNIST -ood COVID-19 --pretrained_path MERes18_PathMNIST_0113_015355 -gid 0
python3 main.py -id PathMNIST -ood COVID-19 --pretrained_path MERes50_PathMNIST_0113_104937 -gid 0

python3 main.py -id ISIC -ood RSNA --pretrained_path MERes18_ISIC_1231_003419 -gid 0
python3 main.py -id ISIC -ood RSNA --pretrained_path MERes50_ISIC_0114_155038 -gid 0
python3 main.py -id PathMNIST -ood RSNA --pretrained_path MERes18_PathMNIST_0113_015355 -gid 0
python3 main.py -id PathMNIST -ood RSNA --pretrained_path MERes50_PathMNIST_0113_104937 -gid 0
