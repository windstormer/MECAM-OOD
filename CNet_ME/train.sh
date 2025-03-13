#! /bin/bash

python3 train_cnet.py -b 128 -d ISIC --model_type Res18 -e 200
python3 train_cnet.py -b 128 -d ISIC --model_type Res50 -e 200

python3 train_cnet.py -b 256 -d PathMNIST --model_type Res18 -e 20
python3 train_cnet.py -b 128 -d PathMNIST --model_type Res50 -e 20
