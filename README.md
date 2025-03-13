# Unsupervised Out-of-Distribution Detection in Medical Imaging Using Multi-Exit Class Activation Maps and Feature Masking (MECAM-OOD)

## Dataset

### ID Dataset - ISIC19, PathMNIST
- [ISIC 2019](https://challenge.isic-archive.com/data/#2019)
- PathMNIST (from medmnist import PathMNIST)

```
DATASET_NAME
|-- train
|   |-- class_1
|   |   |-- xxx.jpg
|   |   |-- ...
|   |-- class_2
|   |-- class_3
|   |-- ...
|-- test
|   |-- class_1
|   |   |-- xxx.jpg
|   |   |-- ...
|   |-- class_2
|   |-- class_3
|   |-- ...
```
### OOD Dataset - RSNA, COVID-19, HeadCT
- [RSNA](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
- [COVID-19](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)
```
DATASET_NAME
|-- class_1
|   |-- xxx.jpg
|   |-- ...
|-- class_2
|-- class_3
|-- ...
```

## Train Classification Model with ID Dataset
```
cd CNet_ME
python3 train_cnet.py -b 128 -d ISIC --model_type Res18 -e 200
```

## Run MECAM-OOD
```
cd MECAM-OODD
python3 main.py -id ISIC -ood RSNA --pretrained_path MERes18_ISIC_1231_003419 -gid 0
```


