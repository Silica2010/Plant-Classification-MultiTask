# -*- coding: utf-8 -*-
"""实验一 单图推理：一次前向输出物种与生长状态"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import PlantMultiTaskModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True, help='待测图片路径')
    ap.add_argument('--weights', type=str, default='best_plant_model.pth')
    ap.add_argument('--img_size', type=int, default=224)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)

    species2idx = ckpt['species2idx']
    status2idx = ckpt['status2idx']
    model = PlantMultiTaskModel(len(species2idx), len(status2idx)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(args.image).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_s, out_st = model(x)
        prob_s = torch.softmax(out_s, dim=1)[0].cpu()
        prob_st = torch.softmax(out_st, dim=1)[0].cpu()

    idx2species = {v: k for k, v in species2idx.items()}
    idx2status = {v: k for k, v in status2idx.items()}
    s_idx = int(prob_s.argmax())
    st_idx = int(prob_st.argmax())

    print('预测植物种类：%s，置信度：%.3f' % (idx2species[s_idx], prob_s[s_idx]))
    print('预测生长状态：%s，置信度：%.3f' % (idx2status[st_idx], prob_st[st_idx]))
    print('\n物种各类概率：')
    for i in range(len(idx2species)):
        print('  %-10s %.3f' % (idx2species[i], prob_s[i]))
    print('状态各类概率：')
    for i in range(len(idx2status)):
        print('  %-10s %.3f' % (idx2status[i], prob_st[i]))


if __name__ == '__main__':
    main()
