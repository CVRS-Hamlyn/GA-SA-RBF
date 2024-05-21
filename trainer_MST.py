from __future__ import absolute_import, division, print_function
from ast import Or
import imp
from math import gamma
from operator import index
from random import sample, shuffle
import time
from tkinter import ON
from turtle import clear, distance, pos, width
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import data_loader
import network
from layers import *
from tqdm.autonotebook import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils.data as D
import time
import json
import copy
from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
import torch.backends.cudnn as cudnn
import seaborn as sns
sns.set(rc={"figure.figsize":(18, 10)}) 



class Trainer:
    def __init__(self, options):
        self.opts = options

        #   self.log_path = os.path.join(self.opts.log_directory, self.opts.model_type + '_' + self.opts.loss_regre)

        #   if not os.path.exists(self.log_path):
            #   os.makedirs(self.log_path)

        #   self.writer = SummaryWriter(self.log_path)

        torch.manual_seed(559)
        torch.cuda.manual_seed(559)
        np.random.seed(559)

        cudnn.benchmark = True
        if not self.opts.no_ema:
            self.ema_loss = ema_losses(init=self.opts.LC_init, decay=self.opts.LC_decay, start_itr=1000, n_steps=self.opts.n_steps)
        self.d_shift = data_shift(n_steps=self.opts.n_steps)

        if self.opts.multi_gpu == True:
            num_gpu = torch.cuda.device_count()
        else:
            num_gpu = 1

        self.path_model = os.path.join(self.opts.model_folder, self.opts.checkpoint_dir)
        self.error_path = os.path.join(self.path_model, 'error')
        self.pred_path = os.path.join(self.path_model, 'pred')
        self.pos_path = os.path.join(self.path_model, 'pos')
        self.BM_path = os.path.join(self.path_model, 'BM')
        if not os.path.exists(self.path_model):
            os.makedirs(self.path_model)
        if not os.path.exists(self.error_path):
            os.makedirs(self.error_path)
        if not os.path.exists(self.pred_path):
            os.makedirs(self.pred_path)
        if not os.path.exists(self.pos_path):
            os.makedirs(self.pos_path)
        if not os.path.exists(self.BM_path):
            os.makedirs(self.BM_path)

	    # self.network_list = {"resnet18": network.resnet18, "resnet34": network.resnet34, "resnet50": network.resnet50, "resnet101": network.resnet101}
        self.network_list = {"sffcnet": network.sffcnet}
        self.models = {}
        self.parameters_to_train = []
        self.results = {}
        
        self.freq_encoding = {}
        
        
        self.writer = SummaryWriter(self.path_model)

        if not self.opts.no_cuda:
            if self.opts.multi_gpu == True:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device('cuda:{}'.format(self.opts.gpu_id) if torch.cuda.is_available() else 'cpu')

        
        self.out_channels = [self.opts.out_channels]
        self.discrete_value = None
        
        train_dataset = data_loader.pcle_dataset('train', self.opts.in_channels,
                                                 self.opts.dis_range, self.opts.ord_regre, self.opts.cls_regre,
                                                 self.discrete_value, self.opts.use_interp, 
                                                 self.opts.mixup, self.opts.norm)
        # train_sampler = DistributedSampler(train_dataset)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.opts.batch_size, shuffle=True,
            num_workers=self.opts.num_workers, pin_memory=True)


        val_dataset = data_loader.pcle_dataset('test', self.opts.in_channels,
                                                 self.opts.dis_range, self.opts.ord_regre, self.opts.cls_regre,
                                                 self.discrete_value, self.opts.use_interp, 
                                                 self.opts.mixup, self.opts.norm)
        self.val_loader = DataLoader(
            val_dataset, 1, False,
            num_workers=self.opts.num_workers, pin_memory=True, drop_last=True)

        self.models["G"] = self.network_list[self.opts.model_type](self.opts.in_channels, self.opts.out_channels)
        self.models["D"] = self.network_list[self.opts.model_type](self.opts.in_channels, self.opts.out_channels, D=True)
        self.models["D"] = D_load_pretrained(self.models["D"], './pre_trained/D.pt')


        if not self.opts.no_multi_step:
            self.models["SeqA"] = network.seq_attn_module(1024, 1, 8, self.opts.n_attn_layers)
  
        num_params_G = count_parameters(self.models["G"])
        num_params_D = count_parameters(self.models["D"])
        if not self.opts.no_multi_step:
            num_params_SeqA = count_parameters(self.models["SeqA"])
            print("Number of Parameters in Sequence Attention: {:.1f}M".format(num_params_SeqA / 1e6))
        print("Number of Parameters in Generator: {:.1f}M".format(num_params_G / 1e6))
        print("Number of Parameters in Discriminitor: {:.1f}M".format(num_params_D / 1e6))
        print("Training model named:\n  ", self.opts.model_type)
        print("Training is using:\n  ", self.device)
        if torch.cuda.is_available():
            print('Using GPU: {} X {}'.format(num_gpu, torch.cuda.get_device_name()))

        print("Checkpoint address:", self.path_model)

        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))
        print("Using RBF encoding:", not (self.opts.no_rbf_enc))
        print("Using Sinusoidal Postional Encoding:", self.opts.PE_sin)
        print("The batch size:", self.opts.batch_size)
        print("Number of Epochs:", self.opts.num_epochs)
        print("Using EMA Regularization:", not self.opts.no_ema)
        if self.opts.ord_regre:
            print("Distance Inference: Linear-Exponential Ordinal Regression {}".format(self.opts.num_disc))
        else:
            print("Distance Inference: {} * tanh(x)".format(self.opts.dis_range))

        print("Number of input channels:", self.opts.in_channels)
        print("Number of output channels:", self.out_channels)
        print("Number of steps:", self.opts.n_steps)
        print("Windoww size of final inference:", self.opts.window_size)
        print("L1 weight:", self.opts.l1_weight)
        # print("Interp loss: {:.1f} * MoI + {:.1f} * SSIM + {:.1f} * BM".format(self.opts.MoI_weight, self.opts.SSIM_weight, self.opts.BM_weight))
        if self.opts.multi_gpu == True:

            self.models["G"] = nn.DataParallel(self.models["G"], device_ids=[0,1,2])
            self.models["D"] = nn.DataParallel(self.models["D"], device_ids=[0,1,2])
            if not self.opts.no_multi_step:
                self.models["SeqA"] = nn.DataParallel(self.models["SeqA"], device_ids=[0,1,2])
            # self.models["infer"] = nn.DataParallel(self.models["infer"], device_ids=[0,1,2])
        
        self.models["G"].to(self.device)
        self.models["D"].to(self.device)
        if not self.opts.no_multi_step:
            self.models["SeqA"].to(self.device)
        # self.models["infer"].to(self.device)
        

        self.parameters_to_train_G = []
        self.parameters_to_train_G += list(self.models["G"].parameters())
        # if not self.opts.no_multi_step:
        self.parameters_to_train_G += list(self.models["SeqA"].parameters())
        # self.parameters_to_train_SeqA_infer += list(self.models["infer"].parameters())
        self.parameters_to_train_D = self.models["D"].parameters()
        # self.parameters_to_train_SeqA  = self.models["SeqA"].parameters()
        self.model_optimizer_G = optim.AdamW(self.parameters_to_train_G, self.opts.learning_rate, weight_decay=self.opts.weight_decay)
        # self.model_optimizer_SeqA= optim.AdamW(self.parameters_to_train_SeqA, self.opts.learning_rate, weight_decay=self.opts.weight_decay)
        self.model_optimizer_D = optim.AdamW(self.parameters_to_train_D, self.opts.learning_rate, weight_decay=self.opts.weight_decay)

        # self.model_optimizer_D = optim.SGD(self.parameters_to_train_D, self.opts.learning_rate, momentum=0.9, weight_decay=0.005)

        self.L1_loss = nn.L1Loss().to(self.device)
        self.Smooth_L1 = nn.SmoothL1Loss().to(self.device)
        self.KL_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.BCE_loss = nn.BCELoss().to(self.device)
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)
        self.MSE = torch.nn.MSELoss().to(self.device)
        # self.ones = Variable(Tensor(self.opts.batch_size, 1).fill_(1.0), requires_grad=False)

        

        
        
        self.model_lr_scheduler_G = optim.lr_scheduler.CyclicLR(self.model_optimizer_G, base_lr=1e-5, max_lr=1e-4, 
                                                            step_size_up=3*((len(train_dataset) // self.opts.batch_size) + 1),
                                                            cycle_momentum=False)

        self.model_lr_scheduler_D = optim.lr_scheduler.CyclicLR(self.model_optimizer_D, base_lr=1e-5, max_lr=1e-4, 
                                                            step_size_up=3*((len(train_dataset) // self.opts.batch_size) + 1),
                                                            cycle_momentum=False)

        self.interpolation = pCLE_interpolation(self.opts.batch_size, self.opts.height, self.opts.width, self.opts.step_size).to(self.device)
        self.e_mask = Grad_Energy().to(self.device)
        self.artifact = torch.from_numpy(np.load("./artifact_mask.npy")) * 1.
        if self.opts.PE_sin == True:
            self.PE_sin = PositionalEncoding1D(1024)


    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.opts.num_epochs):
            self.run_epoch()
    
    def LC_loss(self, out_G, out_R, step_id):
        reg = torch.mean(F.relu(out_R - self.ema_loss.D_fake[step_id]).pow(2)) + torch.mean(F.relu(self.ema_loss.D_real - out_G).pow(2))

        return reg

    def run_epoch(self):
        #   self.model_lr_scheduler.step()
        

        print("Training")
        self.set_train()
        Train_Loader = tqdm(self.train_loader)
        Losses = {}
        Loss_G = []
        Loss_D = []
        Loss_g = []
        Loss_p_s = []
        Loss_p_f = []
        Loss_ex = []
        MAE = []

        for batch_idx, inputs in enumerate(Train_Loader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            self.batch_idx = batch_idx  
            frame = inputs['frame'].to(self.device)
        
            out_G, f_G, d_G, d_infer, d_gt, pos_list = self.Multi_step_training(inputs, frame)
            # input(pos_list)
            
            # true frames
            frame_indexes_R, direction_R, ratio_R = find_nearby_frame_indexes(inputs, inputs['distance'].unsqueeze(-1), self.opts.step_size)
            exist_frames_R = exist_frame_select(inputs, frame_indexes_R)
            frames_R = self.interpolation(exist_frames_R, ratio_R, direction_R)

            frame_R_ch1 = self.normlization(frames_R)
            frame_R_ch2 = self.normlization(frames_R, 8191, -400)
   
            f_R = torch.cat((frame_R_ch1, frame_R_ch2), dim=1)
            out_R = self.models['D'](f_R)


            # loss for G(.)
            self.model_optimizer_G.zero_grad()
            # self.model_optimizer_SeqA.zero_grad()
            Losses['loss_G'] = 0
            Losses['loss_l1'] = 0
            Losses['loss_p_s'] = 0
            Losses['loss_p_f'] = 0
            count = 0
            for i in range(self.opts.n_steps-1, -1, -1):
                Losses['loss_G'] += (self.opts.DC ** count) * (-torch.mean(out_G[i]['fc'])) # hinge
                # Losses['loss_G'] += self.MSE(out_G[i]['fc'], Ones) # LSGAN
                Losses['loss_l1'] += (self.opts.DC ** count) * (self.L1_loss(d_infer[i], d_gt[i]) + self.L1_loss(d_G[i], d_gt[i])) # L1
                # Losses['loss_l1'] += (self.opts.DC ** count) * (self.L1_loss(d_infer[i], d_gt[i]) + self.L1_loss(d_G[i], d_gt[i])) + (self.opts.RDC ** count) * self.opts.alpha * torch.mean(F.relu(-1 * pos_list[i])) # L1
                for j in range(4):
                    Losses['loss_p_s'] += (self.opts.DC ** count) * self.L1_loss(out_G[i]['fmap_s', j], out_R['fmap_s', j].detach())
                    Losses['loss_p_f'] += (self.opts.DC ** count) * self.L1_loss(out_G[i]['fmap_f', j], out_R['fmap_f', j].detach())

                count += 1               
       
            Losses['total_G'] = (Losses['loss_G'] + Losses['loss_p_s'] + Losses['loss_p_f'] + (self.opts.l1_weight * Losses['loss_l1'])) / self.opts.n_steps
            

            Losses['total_G'].backward()
            self.model_optimizer_G.step()
            self.model_lr_scheduler_G.step()
            # self.model_optimizer_SeqA.step()
            self.writer.add_scalar('lr/train_G', np.array(self.model_lr_scheduler_G.get_last_lr()), self.step)

            del out_G
            del d_G
            del d_gt

            # loss for D(.)
            self.model_optimizer_D.zero_grad()

            out_G_seq = {}
            for i in range(self.opts.n_steps):
                out_G_seq[i] = self.models['D'](f_G[i].detach())['fc']             
            out_R = self.models['D'](f_R)['fc']

            if not self.opts.no_ema:
                for i in range(self.opts.n_steps):
                    self.writer.add_scalar('EMA/D_fake_{}'.format(i), self.ema_loss.D_fake[i], self.step)
                    self.ema_loss.update(torch.mean(out_G_seq[i]).item(), 'D_fake', self.step, i)
                self.writer.add_scalar('EMA/D_real', self.ema_loss.D_real, self.step)
                self.ema_loss.update(torch.mean(out_R).item(), 'D_real', self.step)

            Losses['loss_D'] = 0
            count = 0
            for i in range(self.opts.n_steps-1, -1, -1):
                if not self.opts.no_ema and self.step <= self.ema_loss.start_itr:
                    Losses['loss_D'] += (self.opts.DC ** count) * (torch.mean(F.relu(1 + out_G_seq[i])) + torch.mean(F.relu(1- out_R)))# hinge
                elif not self.opts.no_ema and self.step > self.ema_loss.start_itr:
                    Losses['loss_D'] += (self.opts.DC ** count) * (torch.mean(F.relu(1 + out_G_seq[i])) + torch.mean(F.relu(1- out_R)) + self.opts.RC * self.LC_loss(out_G_seq[i], out_R, i)) # hinge + LC
                else:
                    Losses['loss_D'] += (self.opts.DC ** count) * (torch.mean(F.relu(1 + out_G_seq[i])) + torch.mean(F.relu(1- out_R))) # hinge
                # Losses['loss_D'] += 0.5 * (self.MSE(out_R, Ones) + self.MSE(out_G_seq[i], (Ones - ((self.opts.DC) ** i) * Ones))) # LSGAN
                count += 1
            Losses['loss_D'] /= self.opts.n_steps

            Losses['loss_D'].backward()
            self.model_optimizer_D.step()
            self.model_lr_scheduler_D.step()
            self.writer.add_scalar('lr/train_D', np.array(self.model_lr_scheduler_D.get_last_lr()), self.step)

            del out_G_seq
            del out_R

            Loss_G.append(Losses['total_G'].cpu().detach().numpy())
            Loss_D.append(Losses['loss_D'].cpu().detach().numpy())
            MAE.append(Losses['loss_l1'].cpu().detach().numpy())
            Loss_g.append(Losses['loss_G'].cpu().detach().numpy())
            Loss_p_s.append(Losses['loss_p_s'].cpu().detach().numpy())
            Loss_p_f.append(Losses['loss_p_f'].cpu().detach().numpy())
            # Loss_ex.append(Losses['loss_Ex'].cpu().detach().numpy())

            Train_Loader.set_postfix(loss_G=np.mean(Loss_G), loss_D=np.mean(Loss_D), MAE=np.mean(MAE),epoch=self.epoch)

            self.step += 1

        # log training results per epoch

        self.writer.add_scalar('Loss_Generator/train_g', np.mean(Loss_g), self.epoch)
        self.writer.add_scalar('Loss_Generator/train_p_s', np.mean(Loss_p_s), self.epoch)
        self.writer.add_scalar('Loss_Generator/train_p_f', np.mean(Loss_p_f), self.epoch)
        self.writer.add_scalar('Loss_Discriminator/train_d', np.mean(Loss_D), self.epoch)
        # self.writer.add_scalar('Loss_Generator/train_excess', np.mean(Loss_ex), self.epoch)
        self.writer.add_scalar('MAE/train_MAE', np.mean(MAE), self.epoch)
        
        # print("Testing")
        # self.val()
        if self.epoch % 10 == 0 or self.epoch == (self.opts.num_epochs - 1):
            print("Evaluation Metrics")
            self.test()


        self.save_checkpoint()
    
    def Multi_step_training(self, inputs, frame_nn):

        d_G = {}
        d_infer = {}
        d_gt = {}
        f_G = {}
        out_G = {}
        pos_list = {}
        Fvec_seq = []
        pred_dist = []

        index_curr = inputs['index']
        distance_gt = inputs['distance']
        d_gt[0] = distance_gt

        self.d_shift.update('GT', torch.mean(d_gt[0]).item(), self.step, 0)
        self.writer.add_scalar('data_shift/GT_{}'.format(0), self.d_shift.d_shift[0], self.step)
        for i in range(self.opts.n_steps):
            #fake frames
            frames_G, frame_nn, distance, distance_G, index_curr, Fvec_seq, pred_dist = self.Generator(inputs, frame_nn, index_curr, Fvec_seq, pred_dist, i)
            d_G[i] = distance_G
            d_infer[i] = distance
            pos_list[i] = distance_gt - distance
            distance_gt = distance_gt - distance.detach()
            

            dis_sign = torch.sign(distance_gt)
            num_steps = torch.div(torch.abs(distance_gt), 5, rounding_mode='trunc')
            residual_dis = (torch.abs(distance_gt) % 5)

            distance_gt = dis_sign * num_steps * 5
            distance_gt[residual_dis > 2.5] += 5 
            d_gt[i+1] = distance_gt
            if i+1 < self.opts.n_steps:
                self.d_shift.update('GT', torch.mean(d_gt[i+1]).item(), self.step, i+1)
                self.writer.add_scalar('data_shift/GT_{}'.format(i+1), self.d_shift.d_shift[i+1], self.step)
            self.d_shift.update('pred', torch.mean(d_infer[i]).item(), self.step, i)
            self.writer.add_scalar('data_shift/pred_{}'.format(i), self.d_shift.pred_shift[i], self.step)
        
            #generate input data
            frame = torch.cat((self.normlization(frames_G), self.normlization(frames_G, 8191, -400)), dim=1)
            f_G[i] = frame
            
            frame_nn = torch.cat((self.normlization(frame_nn), self.normlization(frame_nn, 8191, -400)), dim=1)

            #generate output of D(.) for G(.)
            out_G[i] = self.models['D'](frame)
            

        return out_G, f_G, d_G, d_infer, d_gt, pos_list

    def Generator(self, inputs, frame, index_curr, Fvec_seq, pred_dist, step_id):
        outputs = self.models['G'](frame)
        Fvec_seq.append(outputs['fvec'])
        
        if step_id == 0:
            distance = torch.tanh(outputs['fc']) * self.opts.dis_range
            distance_G = distance
            pred_dist.append(distance_G.unsqueeze(-1))
        else:
            distance_G = torch.tanh(outputs['fc']) * self.opts.dis_range
            pred_dist.append(distance_G.unsqueeze(-1))
            fvec_seq = torch.cat(Fvec_seq[-self.opts.window_size:], dim=1)
            dist_seq = torch.cat(pred_dist[-self.opts.window_size:], dim=1)
            if self.opts.no_rbf_enc:
                if self.opts.PE_sin == False:
                    dis_enc_c = None
                    dis_enc_seq = None
                else:
                    dis_enc_c = self.PE_sin(distance_G.unsqueeze(-1))
                    dis_enc_seq = self.PE_sin(dist_seq)
            else:
                dis_enc_c = RBF_Distance_Encoding(1024, distance_G.unsqueeze(-1))
                dis_enc_seq = RBF_Distance_Encoding(1024, dist_seq)
            f_o = self.models['SeqA'](outputs['fvec'], fvec_seq, dis_enc_c=dis_enc_c, dis_enc_seq=dis_enc_seq)       
            distance = torch.tanh(f_o) * self.opts.dis_range


        dis_sign = torch.sign(distance).squeeze(-1)
        step_incre = torch.div(torch.abs(distance), 5, rounding_mode='trunc').squeeze(-1)
        residual_dis = (torch.abs(distance) % 5).squeeze(-1)

        step_incre[residual_dis > 2.5] += 1

        index_next = (index_curr + dis_sign * step_incre)
        index_next = torch.maximum(torch.minimum(index_next, inputs['num_frames'] - 1), torch.tensor([0]).to(device=index_next.device))
        index_next = index_next.detach().long()
        b = torch.arange(index_next.shape[0]).long()
        frame_nn = torch.cat([inputs['video'][x,y,:,:].unsqueeze(0).unsqueeze(0) for x, y in zip(b, index_next)])

        frame_indexes, direction, ratio = find_nearby_frame_indexes(inputs, distance, self.opts.step_size)
        exist_frames = exist_frame_select(inputs, frame_indexes)
        interp_frames = self.interpolation(exist_frames, ratio, direction)

        return interp_frames, frame_nn, distance.squeeze(-1), distance_G.squeeze(-1), index_next, Fvec_seq, pred_dist
    

    def norm(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)

        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm

        
    def test(self):

        Acc = []
        MAE = []
        Dis = []
        Steps = []
        self.dist_error = {}
        self.distance_pre = {}
        self.converg_pos = {}
        self.BM = {}
        for i in range(-400, 401, 5):
            self.dist_error[i] = []
            self.distance_pre[i] = []
            self.converg_pos[i] = {}
            self.BM[i] = {}

        self.set_eval()
        Test_Loader = tqdm(self.val_loader)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(Test_Loader):
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                Fvec_seq = []
                pred_dist = []
                frame = inputs['frame']
                video = inputs['video']
                index_curr = inputs['index']
                BM = inputs['BM'][0]
                outputs = self.models['G'](frame)
                # Fvec_seq.append(outputs['fvec'])
                distance = torch.tanh(outputs['fc']) * self.opts.dis_range
                # if not self.opts.no_multi_step:
                #     pred_dist.append(distance.unsqueeze(0))
                dist_target = inputs['distance']
                count = 0
                id = len(self.converg_pos[int(dist_target.cpu().detach().numpy())])
                self.converg_pos[int(dist_target.cpu().detach().numpy())][id + 1] = []
                track_folder = os.path.join(self.path_model, "img", "{}".format(dist_target.cpu().detach().numpy()[0]), "{}".format(id + 1))
                if not os.path.exists(track_folder):
                    os.makedirs(track_folder)
                if self.opts.in_channels == 3:
                    im = self.norm(frame[:,1,:,:], I_max=8191, I_min=-400)
                    plt.imsave(os.path.join(track_folder, "0.png"), im[0,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                else:
                    plt.imsave(os.path.join(track_folder, "0.png"), frame[0,1,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                self.BM[int(dist_target.cpu().detach().numpy())][id + 1] = []
                # print(BM[index_curr.cpu().detach().numpy()[0]].cpu().detach().numpy())
                self.BM[int(dist_target.cpu().detach().numpy())][id + 1].append(float(BM[index_curr.cpu().detach().numpy()[0]].cpu().detach().numpy()))
                for step in range(1,21,1):
                    outputs = self.models['G'](frame)
                    Fvec_seq.append(outputs['fvec'])
                    if step == 1:
                        dist = torch.tanh(outputs['fc']) * self.opts.dis_range
                        pred_dist.append(dist.unsqueeze(0))
                    else:
                        dist_G = torch.tanh(outputs['fc']) * self.opts.dis_range
                        pred_dist.append(dist_G.unsqueeze(0))
                        dist_seq = torch.cat(pred_dist[-self.opts.window_size:], dim=1)
                        if self.opts.no_rbf_enc:
                            if self.opts.PE_sin == False:
                                dis_enc_c = None
                                dis_enc_seq = None
                            else:
                                dis_enc_c = self.PE_sin(dist_G.unsqueeze(-1))
                                dis_enc_seq = self.PE_sin(dist_seq)
                        else:
                            dis_enc_c = RBF_Distance_Encoding(1024, dist_G.unsqueeze(-1))
                            dis_enc_seq = RBF_Distance_Encoding(1024, dist_seq)
                        fvec_seq = torch.cat(Fvec_seq[-self.opts.window_size:], dim=1)
                        f_o = self.models['SeqA'](fvec_c=outputs['fvec'], fvec_seq=fvec_seq, dis_enc_c=dis_enc_c, dis_enc_seq=dis_enc_seq)
                        dist = torch.tanh(f_o) * self.opts.dis_range

                    dis_sign = torch.sign(dist)
                    step_incre = torch.div(torch.abs(dist), 5, rounding_mode='trunc').squeeze(-1)
                    residual_dis = (torch.abs(dist) % 5).squeeze(-1)
                    if step == 1:
                        loc_probe  = dist_target - dist
                    else:
                        loc_probe -= dist
                    
                    self.converg_pos[int(dist_target.cpu().detach().numpy())][id + 1].append(float(loc_probe.cpu().detach().numpy()[0][0]))

                    if residual_dis <= 2.5:
                        step_incre = step_incre
                    else:
                        step_incre += 1
                    if step == 1:
                        index_next = (index_curr + dis_sign * step_incre)
                    else:
                        index_next = (index_next + dis_sign * step_incre)
                    index_next = torch.maximum(torch.minimum(index_next, inputs['num_frames'].unsqueeze(-1) - 1), torch.tensor([0]).to(device=index_next.device))
                    index_next = int(index_next.cpu().detach().numpy()[0])
                    frame_next = video[:,index_next,:,:].unsqueeze(1)
                    self.BM[int(dist_target.cpu().detach().numpy())][id + 1].append(float(BM[index_next].cpu().detach().numpy()))

                    if self.opts.in_channels == 2:
                        ipt_ch1 = self.norm(frame_next, None, None)
                        ipt_ch2 = self.norm(frame_next, I_max=8191, I_min=-400)
                        frame = torch.cat((ipt_ch1, ipt_ch2), dim=1)
                    elif self.opts.in_channels == 3:
                        frame = self.norm(frame_next, None, None).repeat(1, 3, 1, 1)


                    if self.opts.in_channels == 3:
                        im = self.norm(frame_next, I_max=8191, I_min=-400)
                        # input(im.shape)
                        plt.imsave(os.path.join(track_folder, "{}.png".format(step)), im[0,0,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                    else:
                        plt.imsave(os.path.join(track_folder, "{}.png".format(step)), frame[0,1,:,:].cpu().detach().numpy(), cmap='gray',vmin=0,vmax=1)
                    
                #   sign = torch.where(sign == 0, torch.tensor(1).to(self.device), sign)
                mae = self.L1_loss(distance.squeeze(-1), inputs['distance'])
                
                dist_target = inputs['distance'][0]
                self.dist_error[int(dist_target)].append(torch.abs(distance.squeeze(-1)[0] - dist_target).cpu().detach().numpy())
                self.distance_pre[int(dist_target)].append(distance.squeeze(-1)[0].cpu().detach().numpy())
                # mae = torch.where(torch.abs(mae) <= 25, torch.tensor(0,dtype=torch.float64).to(self.device), mae - 25)
                MAE.append(mae.cpu().detach().numpy())
                Dis.append((inputs['distance'] - distance.squeeze(-1)).cpu().detach().numpy())
                Test_Loader.set_postfix(MAE_test = np.mean(MAE),epoch=self.epoch)

            mean_MAE = np.mean(MAE)
            # print("The loaded checkpoint:", self.path_model)
            print("MAE:", mean_MAE)
            np.save(os.path.join(self.error_path, "{}.npy".format(self.epoch)), self.dist_error)
            np.save(os.path.join(self.pred_path, "{}.npy".format(self.epoch)), self.distance_pre)
            with open(os.path.join(self.pos_path, "{}.json".format(self.epoch)), 'w') as f_test:
                json.dump(self.converg_pos, f_test)
            with open(os.path.join(self.BM_path, "{}.json".format(self.epoch)), 'w') as f_BM_test:
                json.dump(self.BM, f_BM_test)
            
            df = pd.Series(self.dist_error).rename_axis('distance').reset_index(name='error')
            df = pd.merge(df, pd.Series(self.distance_pre).rename_axis('distance').reset_index(name='pred_dist'), on='distance')
            df['pred_mean'] = df.pred_dist.apply(lambda x: np.mean(np.array(x)))
            df['pred_median'] = df.pred_dist.apply(lambda x: np.median(np.array(x)))
            df['pred_std'] = df.pred_dist.apply(lambda x: np.std(np.array(x)))
            df['pred_n'] = df.pred_dist.apply(lambda x:(np.array(x).shape[0]))
            df['pred_standard_error'] = df['pred_std']/np.sqrt(df.pred_n)
            df['vector'] = 0-df['pred_mean']

            figure_error = self.plt_error_figure(df=df)
            self.writer.add_figure('figs/pred_to_GT', figure_error, self.epoch)
            dir_acc = self.compute_dir_acc(df=df)
            
            figure_converg, width, BM_SC, E_c = self.plt_convergence()
            self.writer.add_figure('figs/convergence', figure_converg, self.epoch)
            self.writer.add_scalar('Eval_Metrics/dir_acc', dir_acc, self.epoch)
            self.writer.add_scalar('Eval_Metrics/width', width, self.epoch)
            self.writer.add_scalar('Eval_Metrics/BM_SC', BM_SC, self.epoch)
            self.writer.add_scalar('Eval_Metrics/MAE_C', E_c, self.epoch)
            self.writer.add_scalar('Eval_Metrics/MAE_1st', mean_MAE, self.epoch)

            # self.results['test_mae'] = mean_MAE
            # 
        self.set_train()


    def plt_error_figure(self, df):
        fig = plt.figure()
        barlist = plt.bar(
            df.distance.values, df.vector.values, color='blue'
        )
        for bar in barlist:
            bar.set_color('b')
        plt.fill_between(df.distance.values, df.vector.values - df.pred_standard_error, df.vector.values + df.pred_standard_error,
                    color='gray', alpha=0.5
                        )
        plt.title("Predicted focusing distance against GT distance (Ours)", fontsize=24)
        plt.axvline(x=-35, ymin=0, color='green', linewidth=5)
        plt.axvline(x=35, ymin=0, color='green', linewidth=5, label='working range')
        plt.ylabel('Steps to move during Robotic Manipulation',fontsize=18)
        plt.xlabel('GT distance',fontsize=18)

        plt.xlim=(-400, 400)
        plt.ylim=(-400, 400)
        x1, y1 = [-400, 400], [+400, -400]
        plt.plot(x1, y1, marker = 'o', color='r',label='ground truth', linewidth=5)
        plt.legend(fontsize=18)

        return fig

    def compute_dir_acc(self, df):
        # count from above
        count = 0

        # negative side count
        for i in range(80):
            distance_value = df.distance.values[i]
            vector_value = df.pred_dist[i]
            for j in range(len(vector_value)):
                if -1 * vector_value[j]>=0:
                    count+=1
                
        # positive side count
        for i in range(80, len(df.distance.values), 1):
            distance_value = df.distance.values[i]
            vector_value = df.pred_dist[i]
            for j in range(len(vector_value)):
                if -1 * vector_value[j] <= 0:
                    count+=1
        return count / 1706
    
    def plt_convergence(self):
        B_up = []
        B_down = []
        Pos = []
        BM_score = []
        fig = plt.figure()
        plt.xlim=(0, 30)
        plt.ylim=(-400, 400)
        for n, key in enumerate(self.converg_pos.keys()):
        #     if (int(key) >= -35) and (int(key) <= 35):
            dis_in = np.zeros((len(self.converg_pos[key]), 21))
            BM_in = np.zeros((len(self.BM[key]), 21))
            for i in range(len(self.converg_pos[key])):
                temp = self.converg_pos[key][i+1]
                temp_BM = self.BM[key][i+1]
                for j in range(21):
                    if j == 0:
                        dis_in[i,j] = int(key)
                        BM_in[i,j] = temp_BM[j-1]
                    else:
                        dis_in[i,j] = temp[j-1]
                        BM_in[i,j] = temp_BM[j-1]
            Pos.append(dis_in)
            BM_score.append(BM_in)
            med = np.median(dis_in, axis=0)
            std_error = np.std(dis_in, axis=0) / np.sqrt(dis_in.shape[0])
            plt.plot(med)
            B_up.append(med +std_error)
            B_down.append(med - std_error)
            plt.fill_between(range(21), med - std_error, med +std_error, alpha=0.2)
        B_up = np.array(B_up)
        B_down = np.array(B_down)
        max_b_up = np.max(B_up[:, 15:])
        max_b_down = np.min(B_down[:, 15:])
        width = max_b_up - max_b_down

        loc = np.concatenate(Pos)
        score = np.concatenate(BM_score)
        later_pos = loc[:, 15:]
        later_score = score[:, 15:]
        BM_SC_mu = np.mean(later_score, axis=0)
        BM_SC_std = np.std(BM_SC_mu)
        print("Blur Metric values: {:.3f} -/+ {:.3f}".format(np.mean(BM_SC_mu), BM_SC_std))

        mu_pos = np.mean(np.abs(later_pos),axis=0)
        print("Convergence Error: {:.2f} -/+ {:.2f}".format(np.mean(mu_pos), np.std(mu_pos)))
        # print([385]+pos['385']['1'])
        plt.axhline(y=-35, xmin=0, color='green', linewidth=5)
        plt.axhline(y=35, xmin=0, color='green', linewidth=5, label='working range')
        plt.axhline(y=max_b_up, xmin=0, color='red', linestyle='dotted', linewidth=5)
        plt.axhline(y=max_b_down, xmin=0, color='red', linestyle='dotted', linewidth=5, label='upper-lower bound')
        plt.fill_between(range(15,21, 1), np.ones((6))*max_b_down, np.ones((6))*max_b_up, color='red', alpha=0.4)
        plt.title("Convergence and Upper-Lower Bounds (Ours)", fontsize=24)
        plt.ylabel('The Position of pCLE',fontsize=18)
        plt.xlabel('Steps',fontsize=18)
        plt.legend(fontsize=18)
        
        return fig, width, np.mean(BM_SC_mu), np.mean(mu_pos)



    def normlization(self, data, I_max=None, I_min=None):
        if I_max is not None:
            data_norm = (data - I_min) / (I_max - I_min)
        else:
            I_max = torch.amax(data, dim=(1,2,3))
            I_min = torch.amin(data, dim=(1,2,3))
            data_norm = (data - I_min.view(-1, 1, 1, 1)) / (I_max.view(-1, 1, 1, 1) - I_min.view(-1, 1, 1, 1))

        return data_norm

    def loc_regression_loss(self, loc_mask, distance_t):
        distance_t = distance_t.unsqueeze(-1).repeat(1, self.nodes_layers[self.opts.tree_depth])
        loc_mask = loc_mask * 1.
        loss_loc = (loc_mask * self.dis_loc - loc_mask * distance_t) ** 2
        # input(loss_loc)
        return (loss_loc.sum(-1)).mean()



    def save_checkpoint(self):
        PATH = os.path.join(self.path_model, ('model_{}.pt').format(self.epoch))

        torch.save({
                    'epoch': self.epoch,
                    'model_state_dict_G': self.models['G'].state_dict(),
                    'model_state_dict_D': self.models['D'].state_dict(),
                    'model_state_dict_SeqA': self.models['SeqA'].state_dict(),
                    'optimizer_state_dict_G': self.model_optimizer_G.state_dict(),
                    'optimizer_state_dict_D': self.model_optimizer_D.state_dict(),
                    }, PATH)


    def load_checkpoint(self, epoch):
        checkpoint = torch.load(os.path.join(self.path_model, ('model_{}.pt').format(epoch)))
        # input(checkpoint['model_state_dict'].keys())
        for key in list(checkpoint['model_state_dict_G'].keys()):
            checkpoint['model_state_dict_G'][key.replace('module.', '')] = checkpoint['model_state_dict_G'].pop(key)
        self.models['G'].load_state_dict(checkpoint['model_state_dict_G'])
        for key in list(checkpoint['model_state_dict_SeqA'].keys()):
            checkpoint['model_state_dict_SeqA'][key.replace('module.', '')] = checkpoint['model_state_dict_SeqA'].pop(key)
        self.models['SeqA'].load_state_dict(checkpoint['model_state_dict_SeqA'])
        # self.model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # input(checkpoint['model_state_dict'].keys())
        # print('train_MAE', checkpoint['train_MAE'])
        # print('val_MAE', checkpoint['val_MAE'])

    