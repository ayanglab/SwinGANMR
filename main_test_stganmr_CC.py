import argparse
import cv2
import csv
import sys
import numpy as np
from collections import OrderedDict
import os
import torch
from utils import utils_option as option
from torch.utils.data import DataLoader
from models.network_swinir import SwinIR as net
from utils import utils_image as util
# from utils.utils_swinmr import sobel
from data.select_dataset import define_Dataset
import time


def main(json_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # set up model
    if os.path.exists(opt['model_path']):
        print(f"loading model from {opt['model_path']}")
    else:
        print('can\'t find model.')

    model = define_model(opt)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir, border, window_size = setup(opt)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['zf_psnr'] = []
    test_results['zf_ssim'] = []



    with open(os.path.join(save_dir, 'results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR'])

    with open(os.path.join(save_dir, 'results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'FID'])

    with open(os.path.join(save_dir, 'zf_results.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK', 'SSIM', 'PSNR'])

    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'w') as cf:
        writer = csv.writer(cf)
        writer.writerow(['METHOD', 'MASK',
                         'SSIM', 'SSIM_STD',
                         'PSNR', 'PSNR_STD',
                         'FID'])

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    dataset_opt = opt['datasets']['test']

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)

    for idx, test_data in enumerate(test_loader):

        # if idx > 2000:
        #     break

        img_gt = test_data['H'].to(device)
        img_lq = test_data['L'].to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            # h_pad = (h_old // window_size + 1) * window_size - h_old
            # w_pad = (w_old // window_size + 1) * window_size - w_old
            #
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            # img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            #
            # img_gt = torch.cat([img_gt, torch.flip(img_gt, [2])], 2)[:, :, :h_old + h_pad, :]
            # img_gt = torch.cat([img_gt, torch.flip(img_gt, [3])], 3)[:, :, :, :w_old + w_pad]
            time_start = time.time()
            img_gen = model(img_lq)
            time_end = time.time()
            time_c = time_end - time_start  # 运行所花时间
            print('time cost', time_c, 's')

            img_lq = img_lq[..., :h_old * opt['scale'], :w_old * opt['scale']]
            img_gt = img_gt[..., :h_old * opt['scale'], :w_old * opt['scale']]
            img_gen = img_gen[..., :h_old * opt['scale'], :w_old * opt['scale']]

            diff_gen_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_gen)), 10)
            diff_lq_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)


        # save image
        img_lq = img_lq.data.squeeze().float().cpu().numpy()
        img_gt = img_gt.data.squeeze().float().cpu().numpy()
        img_gen = img_gen.data.squeeze().float().cpu().numpy()

        diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        # evaluate psnr/ssim
        psnr = util.calculate_psnr_single(img_gt, img_gen, border=border)
        ssim = util.calculate_ssim_single(img_gt, img_gen, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        print('Testing {:d} - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, psnr, ssim))

        with open(os.path.join(save_dir, 'results.csv'), 'a') as cf:
            writer = csv.writer(cf)
            writer.writerow(['ST-GAN', dataset_opt['mask'], test_results['ssim'][idx], test_results['psnr'][idx]])

        # evaluate psnr/ssim zf
        zf_psnr = util.calculate_psnr_single(img_gt, img_lq, border=border)
        zf_ssim = util.calculate_ssim_single(img_gt, img_lq, border=border)
        test_results['zf_psnr'].append(zf_psnr)
        test_results['zf_ssim'].append(zf_ssim)
        print('ZF Testing {:d} - PSNR: {:.2f} dB; SSIM: {:.4f}; '.format(idx, zf_psnr, zf_ssim))


        with open(os.path.join(save_dir, 'zf_results.csv'), 'a') as cf:
            writer = csv.writer(cf)
            writer.writerow(['ZF', dataset_opt['mask'], test_results['zf_ssim'][idx], test_results['zf_psnr'][idx]])

        img_lq = (np.clip(img_lq, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gt = (np.clip(img_gt, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        img_gen = (np.clip(img_gen, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

        diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8


        isExists = os.path.exists(os.path.join(save_dir, 'ZF'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'ZF'))
        isExists = os.path.exists(os.path.join(save_dir, 'GT'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'GT'))
        isExists = os.path.exists(os.path.join(save_dir, 'Recon'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'Recon'))

        isExists = os.path.exists(os.path.join(save_dir, 'Different'))
        if not isExists:
            os.makedirs(os.path.join(save_dir, 'Different'))

        cv2.imwrite(os.path.join(save_dir, 'ZF', 'ZF_{:05d}.png'.format(idx)), img_lq)
        cv2.imwrite(os.path.join(save_dir, 'GT', 'GT_{:05d}.png'.format(idx)), img_gt)
        cv2.imwrite(os.path.join(save_dir, 'Recon', 'Recon_{:05d}.png'.format(idx)), img_gen)

        diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
        diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(save_dir, 'Different', 'Diff_Recon_{:05d}.png'.format(idx)), diff_gen_x10_color)
        cv2.imwrite(os.path.join(save_dir, 'Different', 'Diff_ZF_{:05d}.png'.format(idx)), diff_lq_x10_color)

    # summarize psnr/ssim

    ave_psnr = np.mean(test_results['psnr'])
    std_psnr = np.std(test_results['psnr'], ddof=1)
    ave_ssim = np.mean(test_results['ssim'])
    std_ssim = np.std(test_results['ssim'], ddof=1)


    print('\n{} \n-- Average PSNR {:.2f} dB ({:.4f} dB)\n-- Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, ave_psnr, std_psnr, ave_ssim, std_ssim))

    # summarize psnr/ssim zf

    zf_ave_psnr = np.mean(test_results['zf_psnr'])
    zf_std_psnr = np.std(test_results['zf_psnr'], ddof=1)
    zf_ave_ssim = np.mean(test_results['zf_ssim'])
    zf_std_ssim = np.std(test_results['zf_ssim'], ddof=1)


    print('\n{} \n-- ZF Average PSNR {:.2f} dB ({:.4f} dB)\n-- ZF Average SSIM  {:.4f} ({:.6f})'
          .format(save_dir, zf_ave_psnr, zf_std_psnr, zf_ave_ssim, zf_std_ssim))


    # FID
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'GT'),
        os.path.join(save_dir, 'Recon'))).read()
    print(log)
    fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['ST-GAN', dataset_opt['mask'],
                         ave_ssim, std_ssim,
                         ave_psnr, std_psnr,
                         fid])
    # FID ZF
    log = os.popen("{} -m pytorch_fid {} {} ".format(
        sys.executable,
        os.path.join(save_dir, 'GT'),
        os.path.join(save_dir, 'ZF'))).read()
    print(log)
    zf_fid = eval(log.replace('FID:  ', ''))

    with open(os.path.join(save_dir, 'zf_results_ave.csv'), 'a') as cf:
        writer = csv.writer(cf)
        writer.writerow(['ZF', dataset_opt['mask'],
                         zf_ave_ssim, zf_std_ssim,
                         zf_ave_psnr, zf_std_psnr,
                         zf_fid])

def define_model(args):

    model = net(upscale=1, in_chans=1, img_size=256, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=args['netG']['embed_dim'], num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load(args['model_path'])
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        
    return model


def setup(args):

    save_dir = f"results/{args['task']}/{args['model_name']}"
    border = 0
    window_size = 8

    return save_dir, border, window_size


if __name__ == '__main__':

    # Test ST-GAN CCnpi
    main(json_path='options/STGAN/test/test_stganmr_CC_sample.json')


