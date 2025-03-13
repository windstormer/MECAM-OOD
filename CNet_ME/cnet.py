import torch
import torch.optim as optim
import os
from datetime import datetime
from collections import OrderedDict
import torch.nn.functional as F

from resnet import *
from tqdm import tqdm
from utils import *
from sklearn.metrics import roc_auc_score

class CNet(object):
    def __init__(self, args, train_loader, val_loader, log_path, record_path, model_name, gpuid):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_path = log_path
        self.num_classes = args.num_classes

        if args.model_type == "Res18":
            model = MEresnet18(num_classes=self.num_classes)
        elif args.model_type == "Res50":
            model = MEresnet50(num_classes=self.num_classes)

        if args.pretrained_path != None:
            pretrained_path = os.path.join(args.project_path, record_path, args.pretrained_path, "model", "model.pth")
            checkpoint = torch.load(pretrained_path, weights_only=False)
            if any(key.startswith('module.') for key in checkpoint.keys()):
                # Strip 'module.' from the keys
                checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
            model.load_state_dict(checkpoint)

        self.lr = args.learning_rate
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.0001)
        if len(gpuid) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpuid)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.model = model.to('cuda')

        self.loss = nn.CrossEntropyLoss()
        self.model_name = model_name
        self.project_path = args.project_path
        self.record_path = record_path

    def run(self):
        log_file = open(self.log_path, "a")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        train_record = {'auc':[], 'loss':[]}
        val_record = {'auc':[], 'loss':[]}
        best_score = 0.0
        for epoch in range(1, self.epochs + 1):
            train_loss, train_auc = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['auc'].append(train_auc)

            self.scheduler.step()
            
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f'Epoch {epoch:4d}/{self.epochs:4d} | Cur lr: {self.scheduler.get_last_lr()[0]} | Train Loss: {train_loss}, Train AUC: {train_auc:.4f}\n')
            log_file.close()
            
            val_loss, val_auc = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['auc'].append(val_auc)
            log_file = open(self.log_path, "a")
            log_file.writelines(
            f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val AUC: {val_auc:.4f}\n")

            cur_score = val_auc
            if cur_score > best_score:
                best_score = cur_score
                log_file.writelines(f"Save model at Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss}, Val AUC: {val_auc:.4f}\n")
                model_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
                torch.save(self.model.state_dict(), model_path)
            log_file.close()

            # parameter_path = os.path.join(self.project_path, self.record_path, self.model_name, "model", "model.pth")
            # torch.save(self.model.state_dict(), parameter_path)
            log_file = open(self.log_path, "a")
            log_file.writelines(str(datetime.now())+"\n")
            log_file.close()
        save_chart(self.epochs, train_record['auc'], val_record['auc'], os.path.join(self.project_path, self.record_path, self.model_name, "auc.png"), name='auc')
        save_chart(self.epochs, train_record['loss'], val_record['loss'], os.path.join(self.project_path, self.record_path, self.model_name, "loss.png"), name='loss')
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint)
        test_loss, test_auc = self.test(self.val_loader)
        log_file = open(self.log_path, "a")
        log_file.writelines(
            f"Final !! Test Loss: {test_loss}, Test AUC: {test_auc:.4f}\n")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels = []
        pred_results = []
        log_file = open(self.log_path, "a")
        

        for _, case_batch, label_batch in train_bar:
            self.optimizer.zero_grad()
            loss, pred_batch, loss_collect = self.step(case_batch, label_batch)
            loss.backward()
            self.optimizer.step()
            total_num += self.batch_size
            total_loss += loss.item() * self.batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
            pred_results.append(pred_batch)
            train_labels.append(label_batch)
            # print(f'Train Loss: {loss_collect}')
            # log_file.writelines(f'Train Loss: {loss_collect}\n')
        log_file.close()

        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        # print(pred_results.shape)
        # print(train_labels.shape)
        auc_score = self.evaluate(train_labels, pred_results)
        return total_loss / total_num, auc_score

    def compute_custom_regularization(self):
        reg_loss = 0.0
        for name, param in self.model.named_parameters():
            if "ic" in name:  # Target custom Conv2d weights
                reg_loss += torch.sum(torch.abs(param) - param)
        return reg_loss

    def step(self, data_batch, label_batch):
        logits_collect, map_collect = self.model(data_batch.cuda())
        loss = 0
        ic_weight = [0.25, 0.5, 0.75, 1.0]
        loss_collect = []
        for idx, logits in enumerate(logits_collect):
            ic_loss = ic_weight[idx] * self.loss(logits, label_batch.squeeze(1).cuda())
            loss_collect.append(ic_loss.detach().cpu())
            loss += ic_loss

        pred = F.softmax(logits, dim=1)

        return loss, pred.detach().cpu(), loss_collect

            
    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for _, case_batch, label_batch in val_bar:
                loss, pred_batch, _ = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                val_bar.set_description('Val Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, self.epochs, total_loss / total_num))
                pred_results.append(pred_batch)
                val_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        auc_score = self.evaluate(val_labels, pred_results)
        return total_loss / total_num, auc_score

    def test(self, loader, load_model=None):
        self.model.eval()
        test_bar = tqdm(loader)
        total_loss, total_num = 0.0, 0
        test_labels = []
        pred_results = []
        with torch.no_grad():
            for _, case_batch, label_batch in test_bar:
                loss, pred_batch, _ = self.step(case_batch, label_batch)

                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                pred_results.append(pred_batch)
                test_labels.append(label_batch)
        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        auc_score = self.evaluate(test_labels, pred_results)
        return total_loss / total_num, auc_score


    def evaluate(self, labels, pred):
        one_hot_labels = np.eye(self.num_classes)[labels.flatten()]
        auc_score = roc_auc_score(one_hot_labels, pred, multi_class='ovr', average='macro')

        return auc_score