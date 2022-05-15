'''
# -----------------------------------------
Model
TES-GAN m.1.1
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G, define_D, define_D_g
from models.model_base import ModelBase
from models.loss import GANLoss, CharbonnierLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_swinmr import *
from utils.utils_Gabor import *
from utils.utils_Sobel import *

class MRI_TESGAN(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt):
        super(MRI_TESGAN, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            self.netD_g = define_D_g(opt)
            self.netD_g = self.model_to_device(self.netD_g)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netD.train()                     # set training mode,for BN
        self.netD_g.train()                   # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'])
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'])
            else:
                print('Copying model for E')
                self.update_E(0)
            self.netE.eval()

        load_path_D = self.opt['path']['pretrained_netD']
        if self.opt['is_train'] and load_path_D is not None:
            print('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, strict=self.opt_train['D_g_param_strict'])
        load_path_D_g = self.opt['path']['pretrained_netD_g']
        if self.opt['is_train'] and load_path_D_g is not None:
            print('Loading model for D_g[{:s}] ...'.format(load_path_D_g))
            self.load_network(load_path_D_g, self.netD_g, strict=self.opt_train['D_g_param_strict'])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
        load_path_optimizerD = self.opt['path']['pretrained_optimizerD']
        if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
            print('Loading optimizerD [{:s}] ...'.format(load_path_optimizerD))
            self.load_optimizer(load_path_optimizerD, self.D_optimizer)
        load_path_optimizerD_g = self.opt['path']['pretrained_optimizerD_g']
        if load_path_optimizerD_g is not None and self.opt_train['D_g_optimizer_reuse']:
            print('Loading optimizerD_g [{:s}] ...'.format(load_path_optimizerD_g))
            self.load_optimizer(load_path_optimizerD_g, self.D_g_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
        self.save_network(self.save_dir, self.netD_g, 'D_g', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_optimizer, 'optimizerD', iter_label)
        if self.opt_train['D_g_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_g_optimizer, 'optimizerD_g', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------

        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.perceptual_lossfn = PerceptualLoss().to(self.device)

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

        self.D_g_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_g_lossfn_weight = self.opt_train['D_g_lossfn_weight']

        self.D_g_update_ratio = self.opt_train['D_g_update_ratio'] if self.opt_train['D_g_update_ratio'] else 1
        self.D_g_init_iters = self.opt_train['D_g_init_iters'] if self.opt_train['D_g_init_iters'] else 0

    def total_loss(self):

        self.alpha = self.opt_train['alpha']
        self.beta = self.opt_train['beta']
        self.gamma = self.opt_train['gamma']

        # H HR, E Recon, L LR
        self.H_k_real, self.H_k_imag = fft_map(self.H)
        self.E_k_real, self.E_k_imag = fft_map(self.E)

        self.loss_image = self.G_lossfn(self.E, self.H)
        self.loss_freq = (self.G_lossfn(self.E_k_real, self.H_k_real) + self.G_lossfn(self.E_k_imag, self.H_k_imag)) / 2
        self.loss_perc = self.perceptual_lossfn(self.E, self.H)

        return self.alpha * self.loss_image + self.beta * self.loss_freq + self.gamma * self.loss_perc

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)
        self.D_g_optimizer = Adam(self.netD_g.parameters(), lr=self.opt_train['D_g_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_g_optimizer,
                                                        self.opt_train['D_g_scheduler_milestones'],
                                                        self.opt_train['D_g_scheduler_gamma']
                                                        ))
    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.H = data['H'].to(self.device)
        self.L = data['L'].to(self.device)
        self.mask = data['mask'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)


    # ----------------------------------------
    # Gabor
    # ----------------------------------------
    def get_gabor(self):
        self.L_gabor = gabor(self.L, self.device)
        self.E_gabor = gabor(self.E, self.device)
        self.H_gabor = gabor(self.H, self.device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netD_g.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        self.netG_forward()
        self.get_gabor()

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first

            pred_g_fake = self.netD(self.E)
            pred_g_fake_g = self.netD_g(self.E_gabor)
            self.loss_adversarial = (self.D_lossfn(pred_g_fake, True) + self.D_g_lossfn(pred_g_fake_g, True))/2
            loss_G_total = self.G_lossfn_weight * self.total_loss() + self.D_lossfn_weight * self.loss_adversarial

            loss_G_total.backward()
            self.G_optimizer.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True
        self.D_optimizer.zero_grad()

        for p in self.netD_g.parameters():
            p.requires_grad = True
        self.D_g_optimizer.zero_grad()
        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        pred_d_real = self.netD(self.H)                # 1) real data
        l_d_real = self.D_lossfn(pred_d_real, True)
        l_d_real.backward()
        # fake
        pred_d_fake = self.netD(self.E.detach().clone()) # 2) fake data, detach to avoid BP to G
        l_d_fake = self.D_lossfn(pred_d_fake, False)
        l_d_fake.backward()

        self.D_optimizer.step()

        # real
        pred_d_real_g = self.netD_g(self.H_gabor)                # 1) real data
        l_d_real_g = self.D_g_lossfn(pred_d_real_g, True)
        l_d_real_g.backward()
        # fake
        pred_d_fake_g = self.netD_g(self.E_gabor.detach().clone()) # 2) fake data, detach to avoid BP to G
        l_d_fake_g = self.D_g_lossfn(pred_d_fake_g, False)
        l_d_fake_g.backward()

        self.D_g_optimizer.step()


        # ------------------------------------
        # record log
        # ------------------------------------
        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
            self.log_dict['G_loss'] = loss_G_total.item()
            self.log_dict['G_loss_image'] = self.loss_image.item()
            self.log_dict['G_loss_frequency'] = self.loss_freq.item()
            self.log_dict['G_loss_preceptual'] = self.loss_perc.item()
            self.log_dict['G_loss_adversarial'] = self.loss_adversarial.item()

        self.log_dict['D_loss_real'] = torch.mean(l_d_real.detach())
        self.log_dict['D_loss_fake'] = torch.mean(l_d_fake.detach())

        self.log_dict['D_g_loss_real'] = torch.mean(l_d_real_g.detach())
        self.log_dict['D_g_loss_fake'] = torch.mean(l_d_fake_g.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
            self.get_gabor()
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
            print(msg)
            msg = self.describe_network(self.netD_g)
            print(msg)


    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        if self.is_train:
            msg += self.describe_network(self.netD)
            msg += self.describe_network(self.netD_g)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
