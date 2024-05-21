from dataclasses import dataclass
from importlib.resources import path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

def D_load_pretrained(model, path):
    model_dict = model.state_dict()
    loaded = torch.load(path)
    loaded['model_state_dict'] = {k: v for k, v in loaded['model_state_dict'].items() if k in model_dict and k[6:8] != 'fc'}
    model_dict.update(loaded['model_state_dict'])
    model.load_state_dict(model_dict)

    return model

def G_load_pretrained(model, path):
    loaded = torch.load(path)
    for key in list(loaded['model_state_dict_G'].keys()):
        loaded['model_state_dict_G'][key.replace('module.', '')] = loaded['model_state_dict_G'].pop(key)
    
    model.load_state_dict(loaded['model_state_dict_G'])

    return model


class data_shift(object):
    def __init__(self, init=400, decay=0.9, start_itr=0, n_steps=2):
        self.d_shift = {}
        self.pred_shift = {}
        for i in range(n_steps):
            self.d_shift[i] = init
            self.pred_shift[i] = init
        self.start_itr = start_itr
        self.decay = decay

    def update(self, mode, cur, itr, step=None):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'GT':
            self.d_shift[step] = self.d_shift[step] * decay + cur*(1 - decay)
        elif mode == 'pred':
            self.pred_shift[step] = self.pred_shift[step] * decay + cur*(1 - decay)

class ema_losses(object):
    def __init__(self, init=1000., decay=0.9, start_itr=0, n_steps=2):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = {}
        for i in range(n_steps):
            self.D_fake[i] = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr, step=None):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake[step] = self.D_fake[step]*decay + cur*(1 - decay)


class OrdinalRegressionLayer(nn.Module):
    def __init__(self, v_linexp, thres, num_ss, dis_range, num_disc):
        super(OrdinalRegressionLayer, self).__init__()

        self.v_linexp = v_linexp
        self.thres = thres
        self.num_ss = num_ss
        self.v_min = torch.tensor(0 + 1)
        self.v_max = torch.tensor(dis_range + 1)
        self.num_disc = num_disc
        self.num_classes = v_linexp.shape[0] - 2
        self.idx_0 = v_linexp.shape[0] // 2
        temp = torch.zeros((1,1))
        temp = F.pad(temp, (self.idx_0 - 1, 0), 'constant', -1)
        self.mask = F.pad(temp, (0, self.idx_0 - 1), 'constant', 1)
        
    def forward(self, x):
        bs, c  = x.size()
        x = x.reshape(bs, 2, -1)
        prob_log = F.log_softmax(x, dim=1).view(bs, c)
        prob = F.softmax(x, dim=1)
        # prob = x[:, :self.num_classes]
        # ratio = x[:, -1]
        # print(self.idx_0)

        self.mask = self.mask.to(x.device)
        self.v_linexp = self.v_linexp.to(x.device)
#         print(self.mask)
#         print(prob.shape)

        pred_labels = torch.sum(((prob[:, 0, :] > 0.5) * 1) * self.mask, dim=1)
        sign = torch.sign(pred_labels)
        labels_abs = torch.abs(pred_labels)

        temp_start= labels_abs - self.thres
        temp_end = labels_abs - self.thres + 1


        v_start = sign * torch.where(temp_start > 0, torch.exp(torch.log(self.v_min) + torch.log(self.v_max / self.v_min) * (temp_start + self.num_ss - 1) / self.num_disc), 5 * (labels_abs) + 1)
        v_end = sign * torch.where(temp_end > 0, torch.exp(torch.log(self.v_min) + torch.log(self.v_max / self.v_min) * (temp_end + self.num_ss - 1) / self.num_disc), 5 * (labels_abs) + 1)
#         print(v_start)
#         print(v_end)
        dist = (v_end + v_start) / 2

        return prob_log, dist.unsqueeze(-1)

def gen_linexp_ord_value(dis_range, num_disc):
        v_min = torch.tensor(0 + 1)
        v_max = torch.tensor(dis_range + 1)
        ids = torch.arange(0, num_disc + 1, 1)

        v_ord_exp = torch.exp(torch.log(v_min) + torch.log(v_max / v_min) * ids / num_disc)
        steps = v_ord_exp[1:] - v_ord_exp[:-1]
        mask = steps < 10
        num_ss = torch.sum(mask) + 1

        v_ord_lin = torch.arange(1, v_ord_exp[num_ss ], 5)
        thres = v_ord_lin.shape[0]
        if v_ord_exp[num_ss] - v_ord_lin[-1] < 5:
            v_ord_exp[num_ss] = v_ord_lin[-1] + 5
            num_ss += 1
            thres += 1
        v_ord_linexp_pos = torch.cat([v_ord_lin, v_ord_exp[num_ss - 1:]])
        
        v_ord_linexp_neg = -1 * torch.flip(v_ord_linexp_pos, [0])

        v_ord_linexp = torch.cat([v_ord_linexp_neg, torch.zeros(1), v_ord_linexp_pos])

        return v_ord_linexp, thres, num_ss

class CosineDecayWithWarmUpScheduler(object):
    def __init__(self,optimizer,base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
        self.optimizer = optimizer
        self.start_warmup_value = start_warmup_value
        self.base_value = base_value
        self.final_value = final_value
        self.warmup_iters = warmup_epochs * niter_per_ep
        self.total_iters = epochs * niter_per_ep
        if warmup_steps > 0:
            self.warmup_iters = warmup_steps
        print("Set warmup steps = %d" % self.warmup_iters)

        self.iters = 0
        self.lr_list = []
        self.current_lr = None
            
    def get_last_lr(self):
        return self.current_lr
        
    def step(self):
        self.iters += 1
        if self.iters <= self.warmup_iters:
            lr = self.start_warmup_value + ((self.base_value-self.start_warmup_value) / self.warmup_iters) * self.iters
        else:
            lr = self.final_value + 0.5 * (self.base_value - self.final_value) * (1 + math.cos(math.pi * (self.iters - self.warmup_iters) / (self.total_iters - self.warmup_iters)))
                
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            self.lr_list.append(lr)
            self.current_lr = lr

def FrequencyEncoding2D(embed_dim, h_num, w_num, h, w, scale):

    assert embed_dim == h_num * w_num
    f_x = torch.linspace(0, w / (2 ** (scale + 1)), w_num).unsqueeze(0).repeat(h_num, 1).unsqueeze(0)
    f_y_neg = torch.linspace(0, - h / (2 ** (scale + 1)) + 1, int(h_num / 2))
    f_y_pos = torch.linspace(h / (2 ** (scale + 1)) - 1, 0, int(h_num / 2))
    f_y = torch.cat((f_y_neg, f_y_pos)).unsqueeze(1).repeat(1, w_num).unsqueeze(0)
    f = torch.cat((f_y, f_x), dim=0).reshape(2, -1).unsqueeze(1)
    if (h / (2 ** scale)) % 2 == 0:
        freq_y_neg = torch.linspace(0, - h / (2 ** (scale + 1)) + 1, int(h / (2 ** (scale + 1))))
        freq_y_pos = torch.linspace(h / (2 ** (scale + 1)), 0, int(h / (2 ** (scale + 1))))
        freq_y = torch.cat((freq_y_neg, freq_y_pos)).unsqueeze(1).repeat(1, int(w / (2 ** (scale + 1))) + 1).unsqueeze(0)
    else:
        freq_y_neg = torch.linspace(0, - int(h / (2 ** (scale + 1))), int(h / (2 ** (scale + 1))) + 1)
        freq_y_pos = torch.linspace(int(h / (2 ** (scale + 1))) - 1, 0, int(h / (2 ** (scale + 1))))
        freq_y = torch.cat((freq_y_neg, freq_y_pos)).unsqueeze(1).repeat(1, int(w / (2 ** (scale + 1))) + 1).unsqueeze(0)

    freq_x = torch.linspace(0, w / (2 ** (scale + 1)), int(w / (2 ** (scale + 1))) + 1).unsqueeze(0).repeat(int(h / 2 ** scale), 1).unsqueeze(0)
    freq = torch.cat((freq_y, freq_x), dim=0).reshape(2, -1).unsqueeze(-1)
    a = 4**scale / 100**2
    b = 4**scale / 100**2
    encode = torch.exp(-(a * (freq[0, :, :] - f[0, :, :])**2 + b * (freq[1, :, :] - f[1, :, :] ) ** 2))
    
    return encode

def RBF_Distance_Encoding(embed_dim, distance):
    '''
    distance -> [bs, N, 1]
    c -> [bs, 1, 1024]
    '''
    bs = distance.shape[0]
    c = torch.linspace(-400, 400, embed_dim).unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1).to(distance.device)
    encoding = torch.exp(-(1/200**2) * (distance - c) ** 2)

    return encoding

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, distance):
        """
        :param tensor: A 3d tensor of size (batch_size, x, 1)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(distance.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")


        self.cached_penc = None
        sin_inp_x = distance * self.inv_freq.to(distance.device)
        self.cached_penc = get_emb(sin_inp_x)
        
        return self.cached_penc

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]
    
    
class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node).squeeze(-1)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

#         self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

def generate_parent_graph_path(tree_depth):
    g = Graph()
    nodes_layers = compute_num_bridgenet(tree_depth)
    weight_id = 0
    for i in range(tree_depth):
    #     print("layer id:", i)
        num_nodes = nodes_layers[i]
        total_above = compute_pre_nodes(nodes_layers, i)
        total_above_next = compute_pre_nodes(nodes_layers, i+1)
        for j in range(num_nodes):
            for n in range(3):
                g.add_edge(j + total_above, total_above_next + j * 2 + n, weight_id)
                weight_id += 1

    deci_nodes = compute_pre_nodes(nodes_layers, tree_depth)
    leaf_list = []
    for i in range(nodes_layers[tree_depth]):
        leaf_list.append(deci_nodes + i)
    paths = {}
    idx = 0
    for leaf in leaf_list:
        n = g.get_vertex(leaf)
        paths[idx] = []
    #     for i in range(len(n.get_connections())):
        paths[idx].append([])
        for a, p4 in enumerate(n.get_connections()):
            if a == 1:
                paths[idx].append([])
            paths[idx][-1].append(n.get_weight(p4))
    #         input(paths)
            if tree_depth >= 2:
                for b, p3 in enumerate(p4.get_connections()):
                    if b == 1:
                        temp = copy.deepcopy(paths[idx][-1][:1])
                        paths[idx].append(temp)
                    paths[idx][-1].append(p4.get_weight(p3))
    #                 input(paths)
                    if tree_depth >= 3:
                        for c, p2 in enumerate(p3.get_connections()):
                            if c == 1:
                                temp = copy.deepcopy(paths[idx][-1][:2])
                                paths[idx].append(temp)
                            paths[idx][-1].append(p3.get_weight(p2))
    #                         input(paths)
                            if tree_depth >= 4:
                                for d, p1 in enumerate(p2.get_connections()):
                                    if d == 1:
                                        temp = copy.deepcopy(paths[idx][-1][:3])
                                        paths[idx].append(temp)
                                    paths[idx][-1].append(p2.get_weight(p1))
    #                                 input(paths)
                                    if tree_depth >= 5:
                                        for p0 in p1.get_connections():
                                            paths[idx][-1].append(p1.get_weight(p0))

        idx += 1
    
    return g, paths, nodes_layers

def gate_function(fc_opts, paths, nodes_layers, tree_depth):
    bs = fc_opts.shape[0]
    fc_opts = fc_opts.reshape(bs, -1, 3)
    fc_opts = F.softmax(fc_opts, dim=-1).reshape(bs, -1)
    gate_func = torch.zeros(bs, nodes_layers[tree_depth]).to(fc_opts.device)
    # print(paths)
    # input(gate_func.shape)
    for i in range(len(paths)):
        path_list = paths[i]
        for j in range(len(path_list)):
            edges = torch.index_select(fc_opts, 1, torch.tensor(path_list[j]).to(fc_opts.device))
            gate_func[:, i] += torch.prod(edges, dim=1)
    
    return gate_func

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def logits_to_distance(logits, loc_regre):
    ratio = torch.sigmoid(logits)
    distance = loc_regre[:, :, 0] + ratio * (loc_regre[:, :, 1] - loc_regre[:, :, 0])

    return distance

def infer_distance(dis_loc, gate_func):
    dis_pred = torch.sum(dis_loc * gate_func, dim=-1)

    return  dis_pred

def compute_num_bridgenet(num_layers):
    num_node = 1
    nodes_layers = {}
    nodes_layers[0] = num_node
    for i in range(num_layers):
        num_node = 3 + 2*(num_node - 1)
        nodes_layers[i+1] = num_node
        
    return nodes_layers

def compute_pre_nodes(nodes_layers, num_layers):
    num_deci_nodes = 0
    for i in range(num_layers):
        num_deci_nodes += nodes_layers[i]
    return num_deci_nodes



def exist_frame_select(inputs, frame_indexes):
    """
    :Input inputs: dictionary contains video: Nx201xHxW
    :Input frame_indexes: Nx2
    :return exist_frames: Nx2xHxW
    """
    video = inputs['video']
    bs, c, h, w = video.shape
    index_range = (torch.tensor(range(c)) + 1).to(device=video.device)
    # get index of selected frames
    index_1st = frame_indexes[:,0].unsqueeze(-1)
    index_2nd = frame_indexes[:,1].unsqueeze(-1)
    # one hot encode Nx201 frames
    onehot_1st = torch.where(index_range==index_1st+1, (index_1st+1) / index_range, index_1st * 0)
    onehot_2nd = torch.where(index_range==index_2nd+1, (index_2nd+1) / index_range, index_2nd * 0)
    # select frames by one hot encode * frames
    frame_1st = (video * onehot_1st.view(bs, -1, 1, 1)).sum(1).unsqueeze(1)
    frame_2nd = (video * onehot_2nd.view(bs, -1, 1, 1)).sum(1).unsqueeze(1)
    # concat two frames to be interpolated
    exist_frames = torch.cat((frame_1st, frame_2nd), dim=1)
    return exist_frames


def floor_div(input, K):
    residual = torch.remainder(input, K)
    out = (input - residual) / K
    return out


def find_nearby_frame_indexes(inputs, distance, step_size):
    """
    :Input inputs: dictionary contains optimal: N (index of optimal frame), 
                                       index: N (index of input frames)
    : return: frame_indexes: Nx2
            : direction: Nx1 -1 or 1
            : ratio: Nx1
    """
    input_index = inputs['index'].unsqueeze(-1)

    direction = torch.sign(distance)

    num_indexes = floor_div(torch.abs(distance), step_size)
    residual = torch.abs(distance) % step_size

    ratio = residual / step_size
    # print('input_index:', input_index.shape)
    # print('direction:', direction.shape)
    # print('num_indexes:', num_indexes.shape)
    # input()

    index_pre = torch.maximum(torch.minimum((input_index + direction * num_indexes), inputs['num_frames'].unsqueeze(-1) - 1), torch.tensor([0]).to(device=input_index.device))
    index_post = torch.maximum(torch.minimum(input_index + direction * (num_indexes + 1), inputs['num_frames'].unsqueeze(-1) -1), torch.tensor([0]).to(device=input_index.device))

    frame_indexes = torch.cat((index_pre, index_post), dim=1)

    return frame_indexes, direction, ratio



class pCLE_interpolation(nn.Module):
    def __init__(self, batch_size, height, width, step_size):
        super(pCLE_interpolation, self).__init__()
        # self.bs = batch_size
        self.height = height
        self.width = width
        self.step_size = step_size

    def forward(self, exist_frames, ratio, direction):
        """
        :Input exist_frames: Nx2xHxW
        :Input ratio: Nx1 a scale value
        :Input direction: Nx1 -1 or 1
        :return: interp_frames: Nx1xHxW
        """
        bs = direction.shape[0]
        interp_frames = torch.zeros(bs, 1, self.height, self.width).to(device=exist_frames.device)

        cond_1 = direction > 0
        cond_2 = direction < 0
        cond_3 = direction == 0
        cond_1_frame = cond_1.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, self.height, self.width)
        cond_2_frame = cond_2.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, self.height, self.width)
        cond_3_frame = cond_3.unsqueeze(-1).unsqueeze(-1).repeat(1, 2, self.height, self.width)
        
        interp_frames[cond_1] = ratio[cond_1].view(-1, 1, 1) * exist_frames[cond_1_frame].view(-1, 2, self.height, self.width)[:, 1, :, :] + \
                               (1 - ratio[cond_1].view(-1, 1, 1)) * exist_frames[cond_1_frame].view(-1, 2, self.height, self.width)[:, 0, :, :]
        interp_frames[cond_2] = ratio[cond_2].view(-1, 1, 1) * exist_frames[cond_2_frame].view(-1, 2, self.height, self.width)[:, 0, :, :] + \
                               (1 - ratio[cond_2].view(-1, 1, 1)) * exist_frames[cond_2_frame].view(-1, 2, self.height, self.width)[:, 1, :, :]
        interp_frames[cond_3] = exist_frames[cond_3_frame].view(-1, 2, self.height, self.width)[:, 0, :, :]

        return interp_frames



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, ws):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(ws, 1)
        self.mu_y_pool   = nn.AvgPool2d(ws, 1)
        self.sig_x_pool  = nn.AvgPool2d(ws, 1)
        self.sig_y_pool  = nn.AvgPool2d(ws, 1)
        self.sig_xy_pool = nn.AvgPool2d(ws, 1)

        self.refl = nn.ReflectionPad2d((ws-1) // 2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class BM(nn.Module):
    def __init__(self):
        super(BM,self).__init__()
        self.aver_h = nn.AvgPool2d((1,9), 1)
        self.aver_v = nn.AvgPool2d((9,1), 1)
        self.pad_h = nn.ZeroPad2d((4, 4, 0, 0))
        self.pad_v = nn.ZeroPad2d((0, 0, 4, 4))
    
    def forward(self, x):
        x_h = self.pad_h(x)
        x_v = self.pad_v(x)

        B_hor = self.aver_h(x_h)
        B_ver = self.aver_v(x_v)

        D_F_ver = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        D_F_hor = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])

        D_B_ver = torch.abs(B_ver[:, :, :-1, :] - B_ver[:, :, 1:, :])
        D_B_hor = torch.abs(B_hor[:, :, :, :-1] - B_hor[:, :, :, 1:])

        T_ver = D_F_ver - D_B_ver
        T_hor = D_F_hor - D_B_hor

        V_ver = torch.maximum(T_ver, torch.tensor([0]).to(x.device))
        V_hor = torch.maximum(T_hor, torch.tensor([0]).to(x.device))

        S_V_ver = torch.sum(V_ver[:, :, 1:-1, 1:-1], dim=(-2, -1))
        S_V_hor = torch.sum(V_hor[:, :, 1:-1, 1:-1], dim=(-2, -1))
        
        blur = torch.maximum(S_V_ver, S_V_hor)

        return blur


class Grad_Energy(nn.Module):

    def __init__(self):

        super(Grad_Energy, self).__init__()

        self.pad_h_1 = nn.ReflectionPad2d((0,1,0,0))
        self.pad_v_1 = nn.ReflectionPad2d((0,0,0,1))
        self.pad_h_2 = nn.ReflectionPad2d((0,2,0,0))
        self.pad_v_2 = nn.ReflectionPad2d((0,0,0,2))
        self.pad_h_3 = nn.ReflectionPad2d((0,4,0,0))
        self.pad_v_3 = nn.ReflectionPad2d((0,0,0,4))
        self.pad_h_4 = nn.ReflectionPad2d((0,16,0,0))
        self.pad_v_4 = nn.ReflectionPad2d((0,0,0,16))
        self.pool = nn.MaxPool2d(kernel_size=(9,9), stride=1, padding=4)
        self.pad = nn.ReflectionPad2d((0,0,7,7))

    def forward(self, x, mask):
        mask = mask.to(x.device)
        mask = self.pad(mask.unsqueeze(0).unsqueeze(0))
        x = self.pad(x)

        grad_x_1 = self.pad_h_1(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y_1 = self.pad_v_1(x[:, :, :-1, :] - x[:, :, 1:, :])
        grad_x_2 = self.pad_h_2(x[:, :, :, :-2] - x[:, :, :, 2:])
        grad_y_2 = self.pad_v_2(x[:, :, :-2, :] - x[:, :, 2:, :])
        grad_x_3 = self.pad_h_3(x[:, :, :, :-4] - x[:, :, :, 4:])
        grad_y_3 = self.pad_v_3(x[:, :, :-4, :] - x[:, :, 4:, :])
        grad_x_4 = self.pad_h_4(x[:, :, :, :-16] - x[:, :, :, 16:])
        grad_y_4 = self.pad_v_4(x[:, :, :-16, :] - x[:, :, 16:, :])

        grad_1 = torch.sqrt(grad_x_1 ** 2 + grad_y_1 ** 2)
        grad_2 = torch.sqrt(grad_x_2 ** 2 + grad_y_2 ** 2)
        grad_3 = torch.sqrt(grad_x_3 ** 2 + grad_y_3 ** 2)
        grad_4 = torch.sqrt(grad_x_4 ** 2 + grad_y_4 ** 2)

        grad_map = (grad_1 + grad_2 + grad_3 + grad_4) * mask
     
        grad_map = self.pool(grad_map)
        grad_max = torch.amax(grad_map, dim=(-2,-1), keepdim=True)
        grad_min = torch.amin(grad_map, dim=(-2,-1), keepdim=True)
        grad_map = (grad_map - grad_min) / (grad_max - grad_min)
#         grad_map = grad_map
      
        return grad_map
        

