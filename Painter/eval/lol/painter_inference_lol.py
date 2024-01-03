# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import sys
import os
import warnings
import pickle

import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import models_painter

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
import datetime
import time


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1'):
    # build model
    model = getattr(models_painter, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, tgt, size, model, out_path, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    valid = torch.ones_like(tgt)
    loss, y, mask, encoder_representations = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]

    return output.numpy(), encoder_representations


def myPSNR(tar_img, prd_img):
    imdff = np.clip(prd_img, 0, 1) - np.clip(tar_img, 0, 1)
    rmse = np.sqrt((imdff ** 2).mean())
    ps = 20 * np.log10(1 / rmse)
    return ps


def get_args_parser():
    parser = argparse.ArgumentParser('low-light enhancement', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='100')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--save_pred', action='store_true', help='save predictions',
                        default=False)
    parser.add_argument('--save_reprs', action='store_true', help='save token representations through layers',
                        default=False)
    parser.add_argument('--many_prompts', action='store_true', help='Run each eval with all prompts',
                        default=False)
    parser.add_argument('--prompt_count', type=str, help='how many randomly sampled prompts',
                        default='1')
                                            
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    ckpt_path = args.ckpt_path
    model = args.model
    prompt = args.prompt
    input_size = args.input_size
    many_prompts = args.many_prompts
    save_reprs = args.save_reprs

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = time.strftime("%H-%M-%S")
    dst_dir = os.path.join('models_inference', ckpt_dir.split('/')[-1],
                           "lol_inference_{}_{}_{}".format(ckpt_file, current_date, current_time))

    

    model_painter = prepare_model(ckpt_path, model)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    img_src_dir = "datasets/light_enhance/eval15/low"
    img_path_list = glob.glob(os.path.join(img_src_dir, "*.png"))

    prompt_src_dir = "datasets/light_enhance/our485/low"

    if not many_prompts:
        prompt_path_list = [os.path.join(prompt_src_dir, prompt + ".png")]
    else:
        prompt_path_list = glob.glob(os.path.join(prompt_src_dir, "*.png"))
        prompt_path_list = np.random.choice(prompt_path_list, int(args.prompt_count), replace=False)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    print("output_dir: {}".format(dst_dir))
    if save_reprs:
        pickle_file_path = os.path.join(dst_dir, 'logs.pkl')

    for prompt_img_path in tqdm.tqdm(prompt_path_list):

        img2_path = prompt_img_path
        tgt2_path = prompt_img_path.replace('low', 'high')
        print('prompt: {}'.format(os.path.basename(prompt_img_path)))

        # load the shared prompt image pair
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((input_size, input_size))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path)
        tgt2 = tgt2.resize((input_size, input_size))
        tgt2 = np.array(tgt2) / 255.
        
        model_painter.eval()
        for img_path in tqdm.tqdm(img_path_list):
            """ Load an image """
            img_name = os.path.basename(img_path)
            out_path = os.path.join(dst_dir, img_name)
            img_org = Image.open(img_path).convert("RGB")
            size = img_org.size
            img = img_org.resize((input_size, input_size))
            img = np.array(img) / 255.

            # load gt
            rgb_gt = Image.open(img_path.replace('low', 'high')).convert("RGB")  # irrelevant to prompt-type
            rgb_gt = np.array(rgb_gt) / 255.

            img = np.concatenate((img2, img), axis=0)
            assert img.shape == (input_size * 2, input_size, 3)
            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            tgt = tgt2  # tgt is not available
            tgt = np.concatenate((tgt2, tgt), axis=0)

            assert tgt.shape == (input_size * 2, input_size, 3)
            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            # make random mask reproducible (comment out to make it change)
            torch.manual_seed(2)

            output, representations = run_one_image(img, tgt, size, model_painter, out_path, device)
            rgb_restored = output
            rgb_restored = np.clip(rgb_restored, 0, 1)

            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, channel_axis=-1, data_range=1)

            logs_arr = [os.path.basename(prompt_img_path), img_name,psnr, ssim, "ReprMatrix"]

            print("PSNR:", psnr, ", SSIM:", ssim, img_name, rgb_restored.shape)

            if args.save_pred:
                output = rgb_restored * 255
                output = Image.fromarray(output.astype(np.uint8))
                output.save(out_path)

            record = {
                "Prompt": os.path.basename(prompt_img_path),
                "Query": img_name,
                "PSNR": psnr,
                "SSIM": ssim,
                "ReprMatrix": representations
            }

            if save_reprs:
                with open(pickle_file_path, 'ab') as f:
                    pickle.dump(record, f)