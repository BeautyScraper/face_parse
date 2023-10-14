#!/usr/bin/python
# -*- encoding: utf-8 -*-

from pathlib import Path
from logger import setup_logger
from model import BiSeNet

import torch
import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

model_pth = r"C:\Users\user\Downloads\send\send\face-parsing.PyTorch\res\cp\79999_iter.pth"

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    mask_img = vis_im * 0
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, 14):
        if pi in [ 7, 8, 9]:
            continue
        index = np.where(vis_parsing_anno == pi)

        mask_img[index[0], index[1], :] = [255,255,255]
        vis_parsing_anno_color[index[0], index[1], :] = [255,255,255]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    # import pdb; pdb.set_trace() 
    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, mask_img)
        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def face_mask_gen_prep(input_img, result_file_path):
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(model_pth))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        gen_face_mask(input_img,to_tensor, net, result_file_path)

def gen_face_mask(input_img, to_tensor, net, result_file_path):

    img = Image.open(input_img)
    image = img.resize((512, 512), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    img = img.cuda()
    out = net(img)[0]
    parsing = out.squeeze(0).cpu().numpy().argmax(0)
    save_path = Path(result_file_path) / Path(input_img).name
    vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=str(save_path))


def main(input_img, result_file_path):
    face_mask_gen_prep(input_img, result_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Parsing Evaluation')
    parser.add_argument('--result_dir', type=str, default=r'C:\temp1111', help='Path to save the parsing results')
    parser.add_argument('--input_img', type=str, default=r'C:\dumpinGGrounds\stuff_pg\outputs\Sherlyn', help='Path to the directory containing input images')
    parser.add_argument('--cp', type=str, default='model_final_diss.pth', help='Path to the trained model checkpoint')
    args = parser.parse_args()

    main( args.input_img, args.result_dir)    
    # main(r"C:\temp\sd.jpg", r'C:\temp11')    
    # evaluate(dspth=r'C:\dumpinGGrounds\stuff_pg\outputs\Sherlyn', cp='79999_iter.pth')


