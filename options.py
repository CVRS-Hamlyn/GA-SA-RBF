from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(os.getcwd()) # the directory that options.py resides in

print(file_dir)
data_folder = os.path.join(file_dir, 'to/your/data')
model_folder = os.path.join(file_dir, 'to/your/model/ckpt/path')
class AutoFocusOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="AutoFocus options")

        # PATHS
        self.parser.add_argument("--root_path",
                                 type=str,
                                 help="The root path",
                                 default=os.path.join(data_folder))
        self.parser.add_argument("--model_folder",
                                 type=str,
                                 help="The folder for storing models",
                                 default=model_folder)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="log_book")
        self.parser.add_argument("--checkpoint_dir",
                                 type=str,
                                 help="The checkpoint directory",
                                 default='freq_attn_RBF')
        self.parser.add_argument("--onehot_model_dir",
                                 type=str,
                                 help="trained one-hot encoder model directory",
                                 default='one_hot_encoder')
        # DATA FEATURES
        self.parser.add_argument("--width",
                                 type=int,
                                 help="width of pCLE image",
                                 default=384)
        self.parser.add_argument("--height",
                                 type=int,
                                 help="height of pCLE image",
                                 default=274)
        
        self.parser.add_argument("--step_size",
                                 type=int,
                                 help="step size of data collection",
                                 default=5)
        self.parser.add_argument("--mixup",
                                 help="if set data argumentation mixup",
                                 action="store_true")

        # TRAINING options
        self.parser.add_argument("--in_channels",
				                 type=int,
				                 help="number of input channels",
				                 default=2)
        self.parser.add_argument("--out_channels",
                                 type=int,
                                 help="number of output channels",
                                 default=1)
        self.parser.add_argument("--no_multi_step",
                                 help="whether whether use multi-step training",
                                 action="store_true")
        self.parser.add_argument("--n_steps",
                                 type=int,
                                 help="number of step for sequential training",
                                 default=5)
        self.parser.add_argument("--window_size",
                                 type=int,
                                 help="window size of sequential attention",
                                 default=5)
        self.parser.add_argument("--D_step",
                                 type=int,
                                 help="Multiple steps for discriminator optimization",
                                 default=1)
        self.parser.add_argument("--G_step",
                                 type=int,
                                 help="Multiple steps for generator optimization",
                                 default=1)
        self.parser.add_argument("--DC",
                                 type=float,
                                 help="The decay coefficient for Multi-step training",
                                 default=0.9)
        self.parser.add_argument("--RDC",
                                 type=float,
                                 help="The regularization decay coefficient for Multi-step training",
                                 default=0.8)
        self.parser.add_argument("--alpha",
                                 type=float,
                                 help="The coefficient for regularization term",
                                 default=0.1)
        self.parser.add_argument("--no_ema",
                                 help="Whether use ema regularization",
                                 action="store_true")
        self.parser.add_argument("--RC",
                                 type=float,
                                 help="regularization coefficient for ema",
                                 default=0.1)
        self.parser.add_argument("--LC_init",
                                 type=float,
                                 help="The initial value for ema",
                                 default=1000) 
        self.parser.add_argument("--LC_decay",
                                 type=float,
                                 help="The decay coefficient for ema",
                                 default=0.9)                           
        self.parser.add_argument("--start_itr",
                                 type=int,
                                 help="The starting interation for EMA regularization",
                                 default=1000)
        self.parser.add_argument("--dis_range",
                                 type=int,
                                 help="range of distance for regression",
                                 default=400)
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="The type of model",
                                 default="sffcnet")
        self.parser.add_argument("--ig_ratio",
                                 type=float,
                                 help="The ratio combining individual and global freq feat",
                                 default=0.75)
        self.parser.add_argument("--norm",
                                 help="whether apply normalization to video and image",
                                 action="store_true")
        self.parser.add_argument("--no_rbf_enc",
                                 help="whether use rbf encoding",
                                 action="store_true")
        self.parser.add_argument("--PE_sin",
                                 help="whether whether use Sinusoidal Postional Encoding",
                                 action="store_true")
        self.parser.add_argument("--num_disc",
                                 type=int,
                                 help="number of discrete values within the distance range",
                                 default=25)
        self.parser.add_argument("--use_interp",
                                 help="whether use frame interpolation",
                                 action="store_true")
        self.parser.add_argument("--loss_regre",
                                 type=str,
                                 help="DICE or MAE",
                                 default="MAE")
        self.parser.add_argument("--use_gaussian_infer",
                                 help="whether use gaussian based inference",
                                 action="store_true")
        self.parser.add_argument("--k_fold",
                                 type=int,
                                 help="The kth dataset is used",
                                 default=4)
        self.parser.add_argument("--entropy_mask",
                                 help="whether use entropy mask to weigh the image-level loss",
                                 action="store_true")
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--n_attn_layers",
                                 type=int,
                                 help="the number of cross-attention layers",
                                 default=1)
        self.parser.add_argument("--optim",
                                 type=str,
                                 help="optimizer",
                                 default="adamw")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--one_hot_lr",
                                 type=float,
                                 help="learning rate for one-hot encoder",
                                 default=1)
        self.parser.add_argument("--cyclic_lr",
                                 help="if set use cyclical learning rate",
                                 action="store_true")
        self.parser.add_argument("--cos_lr",
                                 help="if set use cosine learning rate",
                                 action="store_true")
        self.parser.add_argument("--warmup_epochs",
                                 type=int,
                                 help="The number of epochs for warming up",
                                 default=10)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="regularization",
                                 default=1e-2)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=40)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--l1_weight",
                                 type=float,
                                 help="the weight of MAE to balance the loss function",
                                 default=5e-1)
        self.parser.add_argument("--var_weight",
                                 type=float,
                                 help="the weight of variance loss to balance loss function",
                                 default=5e-2)
        self.parser.add_argument("--MoI_weight",
                                 type=float,
                                 help="The weight of MoI loss",
                                 default=1)
        self.parser.add_argument("--local_rank",                        
                                 type=int,
                                 default=2,
                                 help='node rank for distributed training')
        self.parser.add_argument("--SSIM_weight",
                                 type=float,
                                 help="The weight of SSIM loss",
                                 default=1)
        self.parser.add_argument("--BM_weight",
                                 type=float,
                                 help="The weight of blurry metric loss",
                                 default=0)
        self.parser.add_argument("--lamda",
                                 type=float,
                                 help="The weight to balance the contribution",
                                 default=1e-3)
        


        # ABLATION options
        self.parser.add_argument("--pretrained",
                                 help="if set use pretrained weights",
                                 action="store_true")

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=0)
        self.parser.add_argument("--multi_gpu",
                                 help="if set use multi_gpu",
                                 action="store_true")
        self.parser.add_argument("--gpu_id",
                                 type=int,
                                 help="Tthe id of gpu to be used",
                                 default=0)
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


