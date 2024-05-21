
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from .attention import Seq_Atten, MultiheadSeqAttention
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model




def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)



class Fourier_conv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(Fourier_conv, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        bs, c, h ,w = x.size()
        ffted = fft.rfftn(x, s=(h,w), dim=(-2, -1), norm='ortho')
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        
        ffted = torch.split(ffted, int(ffted.shape[1] / 2), dim=1)
        ffted = torch.complex(ffted[0],ffted[1])
        out = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')
        
        return out



class SFFC_Net(nn.Module):
    def __init__(self, FFT_block, in_channels=3, out_channels=1, D=False):
        super(SFFC_Net, self).__init__()
        
        self.D = D
        self.num_ch_model = np.array([64, 64, 128, 256, 512])
        self.model = models.resnet18()
        self.pad = nn.ReflectionPad2d((0,0,7,7))
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
                                nn.Dropout(p=0.2),
                                nn.Linear(512 * 2, 256, bias=True),
                                nn.Linear(256, out_channels, bias=True)
                                )
        self.conv_1x1_rf0 = nn.Conv2d(self.num_ch_model[0], self.num_ch_model[0], kernel_size=1, bias=False)
        self.fft_conv0 = FFT_block(self.num_ch_model[0], self.num_ch_model[0])
        self.conv_1x1_fr0 = nn.Conv2d(self.num_ch_model[0] * 2, self.num_ch_model[0], kernel_size=1, bias=False)
        
        self.conv_1x1_rf1 = nn.Conv2d(self.num_ch_model[0] + self.num_ch_model[1], self.num_ch_model[1], kernel_size=1, bias=False)
        self.fft_conv1 = FFT_block(self.num_ch_model[1], self.num_ch_model[1])
        self.conv_1x1_fr1 = nn.Conv2d(self.num_ch_model[1] + self.num_ch_model[1], self.num_ch_model[1], kernel_size=1, bias=False)
        
        self.conv_1x1_rf2 = nn.Conv2d(self.num_ch_model[1] + self.num_ch_model[2], self.num_ch_model[2], kernel_size=1, bias=False)
        self.fft_conv2 = FFT_block(self.num_ch_model[2], self.num_ch_model[2])
        self.conv_1x1_fr2 = nn.Conv2d(self.num_ch_model[2] + self.num_ch_model[2], self.num_ch_model[2], kernel_size=1, bias=False)
        
        self.conv_1x1_rf3 = nn.Conv2d(self.num_ch_model[2] + self.num_ch_model[3], self.num_ch_model[3], kernel_size=1, bias=False)
        self.fft_conv3 = FFT_block(self.num_ch_model[3], self.num_ch_model[3])
        self.conv_1x1_fr3 = nn.Conv2d(self.num_ch_model[3] + self.num_ch_model[3], self.num_ch_model[3], kernel_size=1, bias=False)
        
        self.conv_1x1_rf4 = nn.Conv2d(self.num_ch_model[3] + self.num_ch_model[4], self.num_ch_model[4], kernel_size=1, bias=False)
        self.fft_conv4 = FFT_block(self.num_ch_model[4], self.num_ch_model[4])
        self.conv_1x1_fr4 = nn.Conv2d(self.num_ch_model[4] + self.num_ch_model[4], self.num_ch_model[4], kernel_size=1, bias=False)
        
        self.down_sample = nn.MaxPool2d((2,2), 2)
        self._reset_parameters()

    def _reset_parameters(self):
        
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        out = {}
        x = self.pad(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        
        fmap_0 = self.model.relu(x)
        fmap_rf_0 = self.conv_1x1_rf0(fmap_0)
        fmap_f_0 = self.fft_conv0(fmap_rf_0)
        fmap_fr_0 = torch.cat((fmap_0, fmap_f_0), dim=1)
        fmap_r_0 = self.conv_1x1_fr0(fmap_fr_0)
        fmap_f_ds_0 = self.down_sample(fmap_f_0)
        
        fmap_1 = self.model.layer1(self.model.maxpool(fmap_r_0))
        fmap_rf_1 = self.conv_1x1_rf1(torch.cat((fmap_f_ds_0, fmap_1), dim=1))
        fmap_f_1 = self.fft_conv1(fmap_rf_1)
        fmap_fr_1 = torch.cat((fmap_1, fmap_f_1), dim=1)
        fmap_r_1 = self.conv_1x1_fr1(fmap_fr_1)
        fmap_f_ds_1 = self.down_sample(fmap_f_1)
        
        fmap_2 = self.model.layer2(fmap_r_1)
        fmap_rf_2 = self.conv_1x1_rf2(torch.cat((fmap_f_ds_1, fmap_2), dim=1))
        fmap_f_2 = self.fft_conv2(fmap_rf_2)
        fmap_fr_2 = torch.cat((fmap_2, fmap_f_2), dim=1)
        fmap_r_2 = self.conv_1x1_fr2(fmap_fr_2)
        fmap_f_ds_2 = self.down_sample(fmap_f_2)
        
        fmap_3 = self.model.layer3(fmap_r_2)
        fmap_rf_3 = self.conv_1x1_rf3(torch.cat((fmap_f_ds_2, fmap_3), dim=1))
        fmap_f_3 = self.fft_conv3(fmap_rf_3)
        fmap_fr_3 = torch.cat((fmap_3, fmap_f_3), dim=1)
        fmap_r_3 = self.conv_1x1_fr3(fmap_fr_3)
        fmap_f_ds_3 = self.down_sample(fmap_f_3)
        
        fmap_4 = self.model.layer4(fmap_r_3)
        fmap_rf_4 = self.conv_1x1_rf4(torch.cat((fmap_f_ds_3, fmap_4), dim=1))
        fmap_f_4 = self.fft_conv4(fmap_rf_4)
        fmap_fr_4 = torch.cat((fmap_4, fmap_f_4), dim=1)
        fmap_r_4 = self.conv_1x1_fr4(fmap_fr_4)
        
        fvec_r = self.model.avgpool(fmap_r_4)
        fvec_f = self.model.avgpool(fmap_f_4)
        
        fvec = torch.cat((fvec_r, fvec_f), dim=1)
        fvec = torch.flatten(fvec, 1)
        
        if self.D:
            out['fmap_s', 0] = fmap_1
            out['fmap_f', 0] = fmap_f_1
            out['fmap_s', 1] = fmap_2
            out['fmap_f', 1] = fmap_f_2
            out['fmap_s', 2] = fmap_3
            out['fmap_f', 2] = fmap_f_3
            out['fmap_s', 3] = fmap_4
            out['fmap_f', 3] = fmap_f_4
            out['fc'] = self.model.fc(fvec)
            return out
        else:
            out['fvec'] = fvec.unsqueeze(1)
            out['fc'] = self.model.fc(fvec)
            return out
        

def sffcnet(in_channels, out_channels, D=False):
    model = SFFC_Net(Fourier_conv, in_channels, out_channels, D=D)

    return model


class Seq_Atten(nn.Module):

    def __init__(self, embed_channels, out_channels, n_heads, n_layers=1):
        super().__init__()
        
        self.mlp_head = nn.Sequential(
                                    nn.LayerNorm(1024),
                                    nn.Linear(1024, out_channels)
                                    )
        self.n_layers = n_layers
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                MultiheadSeqAttention(embed_channels, n_heads)
            )
        self._reset_parameters()
    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, fvec_c, fvec_seq, dis_enc_c=None, dis_enc_seq=None):
        for layer in self.layers:
            f_o = layer(fvec_c, fvec_seq, dis_enc_c, dis_enc_seq)
        out = self.mlp_head(f_o.squeeze(1))
        # if self.n_layers == 1:
        #     f_o = self.seq_attn(fvec_c, fvec_seq, dis_enc_c, dis_enc_seq)
        #     out = self.mlp_head(f_o.squeeze(1))
        # else:
        #     f_o = self.seq_attn(fvec_c, fvec_seq, dis_enc_c, dis_enc_seq)
        #     for i in range(self.n_layers-1):
        #         fvec_seq[-1] = f_o
        #         f_o = self.seq_attn(f_o, fvec_seq, dis_enc_c, dis_enc_seq)
        #     out = self.mlp_head(f_o.squeeze(1))
        return out

def seq_attn_module(embed_channels, out_channels, n_heads, n_layers):
    module = Seq_Atten(embed_channels, out_channels, n_heads, n_layers)

    return module

