# -*- coding: utf-8 -*-
"""实验一 训练脚本：两阶段迁移 + 双分支交叉熵加权"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from model import PlantMultiTaskModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDataset(Dataset):
    """读取 csv：image_path,species,status"""

    def __init__(self, csv_path, label_map, transform=None):
        self.samples = []
        self.transform = transform
        self.species2idx = label_map['species2idx']
        self.status2idx = label_map['status2idx']
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            if header[:3] != ['image_path', 'species', 'status']:
                raise ValueError('csv 列名应为 image_path,species,status，实际为 %s' % header)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, sp, st = line.split(',')[:3]
                self.samples.append((path, self.species2idx[sp], self.status2idx[st]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sp, st = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, sp, st


def build_transform(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def class_weights(dataset, key):
    """按频次的倒数计算类别权重，用于抑制长尾"""
    if key == 'species':
        labels = [s[1] for s in dataset.samples]
    else:
        labels = [s[2] for s in dataset.samples]
    cnt = Counter(labels)
    n_cls = max(labels) + 1
    w = torch.tensor([len(labels) / (n_cls * cnt.get(i, 1)) for i in range(n_cls)],
                     dtype=torch.float)
    return w / w.mean()


@torch.no_grad()
def evaluate(model, loader, device, alpha):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    pred_s_all, true_s_all, pred_st_all, true_st_all = [], [], [], []
    for imgs, sp, st in loader:
        imgs, sp, st = imgs.to(device), sp.to(device), st.to(device)
        out_s, out_st = model(imgs)
        loss = alpha * ce(out_s, sp) + (1 - alpha) * ce(out_st, st)
        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pred_s_all += out_s.argmax(1).cpu().tolist()
        true_s_all += sp.cpu().tolist()
        pred_st_all += out_st.argmax(1).cpu().tolist()
        true_st_all += st.cpu().tolist()
    acc_s = sum(p == t for p, t in zip(pred_s_all, true_s_all)) / n
    acc_st = sum(p == t for p, t in zip(pred_st_all, true_st_all)) / n
    f1_s = f1_score(true_s_all, pred_s_all, average='macro', zero_division=0)
    f1_st = f1_score(true_st_all, pred_st_all, average='macro', zero_division=0)
    return {'loss': loss_sum / n, 'acc_species': acc_s, 'acc_status': acc_st,
            'f1_species': f1_s, 'f1_status': f1_st, 'mean_acc': (acc_s + acc_st) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--labels', type=str, default='configs/labels.json')
    ap.add_argument('--backbone', type=str, default='resnet18')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs_stage1', type=int, default=10)
    ap.add_argument('--epochs_stage2', type=int, default=30)
    ap.add_argument('--lr_stage1', type=float, default=3e-4)
    ap.add_argument('--lr_stage2', type=float, default=1e-5)
    ap.add_argument('--alpha', type=float, default=0.5, help='物种损失权重')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out_dir', type=str, default='runs')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    n_species = len(label_map['species2idx'])
    n_status = len(label_map['status2idx'])

    train_ds = PlantDataset(args.train_csv, label_map, build_transform(args.img_size, True))
    val_ds = PlantDataset(args.val_csv, label_map, build_transform(args.img_size, False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = PlantMultiTaskModel(n_species, n_status, backbone=args.backbone).to(device)
    w_species = class_weights(train_ds, 'species').to(device)
    w_status = class_weights(train_ds, 'status').to(device)
    ce_species = nn.CrossEntropyLoss(weight=w_species)
    ce_status = nn.CrossEntropyLoss(weight=w_status)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best = {'mean_acc': -1.0}
    for stage, (epochs, lr, freeze) in enumerate([
            (args.epochs_stage1, args.lr_stage1, True),
            (args.epochs_stage2, args.lr_stage2, False)], start=1):
        if freeze:
            model.freeze_backbone()
            params = list(model.head_species.parameters()) + list(model.head_status.parameters())
        else:
            model.unfreeze_backbone()
            params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

        for ep in range(1, epochs + 1):
            model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc='stage%d ep%d' % (stage, ep), ncols=100)
            for imgs, sp, st in pbar:
                imgs, sp, st = imgs.to(device), sp.to(device), st.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out_s, out_st = model(imgs)
                    loss = args.alpha * ce_species(out_s, sp) + \
                        (1 - args.alpha) * ce_status(out_st, st)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * imgs.size(0)
                pbar.set_postfix(loss='%.4f' % loss.item())
            scheduler.step()

            metrics = evaluate(model, val_loader, device, args.alpha)
            print('[stage%d ep%02d] train_loss=%.4f val_loss=%.4f acc_s=%.4f acc_st=%.4f '
                  'f1_s=%.4f f1_st=%.4f mean=%.4f'
                  % (stage, ep, running / len(train_ds), metrics['loss'],
                     metrics['acc_species'], metrics['acc_status'],
                     metrics['f1_species'], metrics['f1_status'], metrics['mean_acc']))

            if metrics['mean_acc'] > best['mean_acc']:
                best = dict(metrics)
                best['stage'], best['epoch'] = stage, ep
                torch.save({'model': model.state_dict(),
                            'species2idx': label_map['species2idx'],
                            'status2idx': label_map['status2idx'],
                            'args': vars(args)}, 'best_plant_model.pth')
                print('  -> 保存最优模型 best_plant_model.pth (mean_acc=%.4f)' % best['mean_acc'])

    print('训练结束，最优验证平均准确率 %.4f（stage%d/epoch%d）'
          % (best['mean_acc'], best['stage'], best['epoch']))

    # 训练结束后用最优权重在验证集上给出分类报告，便于观察易混类别
    ckpt = torch.load('best_plant_model.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    pred_s, true_s, pred_st, true_st = [], [], [], []
    with torch.no_grad():
        for imgs, sp, st in val_loader:
            imgs = imgs.to(device)
            out_s, out_st = model(imgs)
            pred_s += out_s.argmax(1).cpu().tolist()
            pred_st += out_st.argmax(1).cpu().tolist()
            true_s += sp.tolist()
            true_st += st.tolist()
    idx2species = {v: k for k, v in label_map['species2idx'].items()}
    idx2status = {v: k for k, v in label_map['status2idx'].items()}
    names = [idx2species[i] for i in range(len(idx2species))]
    print('\n物种分类报告：')
    print(classification_report(true_s, pred_s, target_names=names, zero_division=0))
    print('状态分类报告：')
    print(classification_report(true_st, pred_st,
                                target_names=[idx2status[i] for i in range(len(idx2status))],
                                zero_division=0))


if __name__ == '__main__':
    main()
