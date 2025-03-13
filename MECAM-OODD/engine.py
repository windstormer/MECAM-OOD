import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np

from resnet import *
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import cv2
import torch
from metrics import *
from sklearn.metrics import roc_auc_score, roc_curve
from utils import *


class Engine(object):
    def __init__(self, args, id_loader, ood_loader, record_path, gpuid):

        model_path = os.path.join(args.project_path, record_path, args.pretrained_path, "model", "model.pth")

        model_type, _, date, time = args.pretrained_path.split("_")
        timestamp = f"{date}_{time}"
        self.id_loader = id_loader
        self.ood_loader = ood_loader
        self.args = args
        self.selected_exit = args.selected_exit
        if model_type == "MERes18":
            model = MEresnet18(num_classes=args.num_classes)
        elif model_type == "MERes50":
            model = MEresnet50(num_classes=args.num_classes)

        if args.pretrained_path != None:
            checkpoint = torch.load(model_path, weights_only=False)
            if any(key.startswith('module.') for key in checkpoint.keys()):
                # Strip 'module.' from the keys
                checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
            model.load_state_dict(checkpoint)

        for param in model.parameters():
            param.requires_grad = False

        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        self.model = model.to('cuda')

        exp_name = f"{model_type}_mask_{args.id_dataset}_{args.ood_dataset}_{timestamp}"
        if args.selected_exit != [0,1,2,3]:
            ex = "".join(str(x) for x in args.selected_exit)
            exp_name += f".e{ex}"
        if args.suffix != None:
            exp_name += f".{args.suffix}"

        self.result_path = os.path.join(args.project_path, "results", exp_name)
        if not os.path.exists(os.path.join(self.result_path, "images")):
            os.makedirs(os.path.join(self.result_path, "images"))
        
        result_log_path = os.path.join(self.result_path, "result.log")
        result_log_file = open(result_log_path, "w+")
        result_log_file.writelines(str(datetime.now())+"\n")
        result_log_file.writelines("============ Config =============\n")
        for k, v in vars(args).items():
            print(f"{k}: {v}")
            result_log_file.writelines(f"{k}: {v}\n")
        result_log_file.writelines("=================================\n")
        result_log_file.writelines(f"Test Results:\n")
        result_log_file.close()

        auc, fpr95 = self.test()
    
        result_log_file = open(result_log_path, "a")
        result_log_file.writelines(f"MECAM-OODD: ")
        result_log_file.writelines(f"AUC: {auc:.4f} FPR95: {fpr95:.4f}\n")
        result_log_file.close()


    def step(self, img):
        img = img.cuda()
        logit_collect, map_collect, feature = self.model(img)
        max_idx = torch.argmax(logit_collect, dim=2)
        expanded_indices = max_idx[:, -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        expanded_indices = expanded_indices.expand(-1, 4, -1, 224, 224)
        pred = F.softmax(map_collect, dim=2)
        output = torch.gather(pred, 2, expanded_indices) # select class for each exit
        output = output[:, self.selected_exit]

        conf_class = F.softmax(logit_collect, dim=2)
        conf_class = torch.gather(conf_class, 2, max_idx.unsqueeze(-1))
        
        conf_class = conf_class[:, self.selected_exit]
        weight = (conf_class / torch.sum(conf_class, dim=1, keepdim=True)).unsqueeze(-1).unsqueeze(-1)
        
        map_mask = torch.sum(output*weight, dim=1)
        map_mask = torch.clamp(map_mask, 0, 1)
        masked_img = img * (1-map_mask).cuda()
        _, _, masked_feature = self.model(masked_img)
        return feature, masked_feature, map_mask


    def test(self):
        self.model.eval()

        id_score_list = []
        ood_score_list = []

        score_log_path = os.path.join(self.result_path, f"score.log")
        score_log_file = open(score_log_path, "w+")
        score_log_file.writelines(str(datetime.now())+"\n")
        score_log_file.close()

        id_seen = {}
        test_bar = tqdm(self.id_loader)
        with torch.no_grad():
            for img_path, case_batch, label_batch in test_bar:
                # print(img_path[0], label_batch[0])
                feature, masked_feature, map_mask = self.step(case_batch)
                
                for idx, input_image in enumerate(case_batch):
                    label = label_batch[idx].item()
                    img_name = os.path.splitext(img_path[idx])[0].split(os.path.sep)[-1]
                    # print(img_name)
                    if label not in id_seen:
                        id_seen[label] = 0
                    if id_seen[label] < 10:
                        self.output_CAM(input_image.numpy(), img_name, map_mask[idx].numpy())
                        id_seen[label] += 1
                    ood_score = self.get_ood_score(feature[idx], masked_feature[idx])
                    id_score_list.append(ood_score)
                    score_log_path = os.path.join(self.result_path, f"score.log")
                    score_log_file = open(score_log_path, "a")
                    score_log_file.writelines(f"{img_name}: {ood_score:.4f}\n")
                    score_log_file.close()
        
        ood_seen = {}
        test_bar = tqdm(self.ood_loader)
        with torch.no_grad():
            for img_path, case_batch, label_batch in test_bar:
                feature, masked_feature, map_mask = self.step(case_batch)
                
                for idx, input_image in enumerate(case_batch):
                    label = label_batch[idx].item()
                    img_name = os.path.splitext(img_path[idx])[0].split(os.path.sep)[-1]
                    # print(img_name)
                    if label not in ood_seen:
                        ood_seen[label] = 0
                    if ood_seen[label] < 10:
                        self.output_CAM(input_image.numpy(), img_name, map_mask[idx].numpy())
                        ood_seen[label] += 1
                    ood_score = self.get_ood_score(feature[idx], masked_feature[idx])
                    ood_score_list.append(ood_score)
                    score_log_path = os.path.join(self.result_path, f"score.log")
                    score_log_file = open(score_log_path, "a")
                    score_log_file.writelines(f"{img_name}: {ood_score:.4f}\n")
                    score_log_file.close()

        id_score_list = np.asarray(id_score_list)
        ood_score_list = np.asarray(ood_score_list)

        auc, fpr95 = self.evaluate(id_score_list, ood_score_list)
        plot_roc_curve(id_score_list, ood_score_list, os.path.join(self.result_path, f"roc_curve.png"))
        plot_score_distribution(id_score_list, ood_score_list, os.path.join(self.result_path, f"score_distribution.png"))
        # print(pred_results.shape)
        # print(test_labels.shape)
        return auc, fpr95


    def evaluate(self, ind_score, ood_score):
        auc_score = auc(ind_score, ood_score)[0]
        fpr95, _ = fpr_recall(ind_score, ood_score)
        return auc_score, fpr95

    def heatmap_postprocess(self, feat_map):
        heatmap = cv2.applyColorMap(np.uint8(255 * feat_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        return heatmap

    def img_fusion(self, image, heatmap):
        cam = 0.5*heatmap + 0.5*image
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def output_CAM(self, input_image, img_name, map_mask):
        save_dict_path = os.path.join(self.result_path, "images", img_name)
        if not os.path.exists(save_dict_path):
            os.makedirs(save_dict_path)
        input_image = np.transpose(input_image, (1,2,0))
        io.imsave(os.path.join(save_dict_path, f"input.png"), img_as_ubyte(input_image), check_contrast=False)
        # print(map_mask[0].shape, map_mask[0].min(), map_mask[0].max())
        cam_heat = self.heatmap_postprocess(map_mask[0])
        img_cam_fusion = self.img_fusion(input_image, cam_heat)

        io.imsave(os.path.join(save_dict_path, f"map_mask.png"), img_as_ubyte(map_mask[0]), check_contrast=False)
        io.imsave(os.path.join(save_dict_path, f"map_heat.png"), img_as_ubyte(cam_heat), check_contrast=False)
        io.imsave(os.path.join(save_dict_path, f"fusion_input.png"), img_as_ubyte(img_cam_fusion), check_contrast=False)

    def get_ood_score(self, feature, masked_feature):
        mse = torch.nn.MSELoss()
        score = mse(feature, masked_feature)
        return score
