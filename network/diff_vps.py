import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)
from transformers import SegformerDecodeHead
from network.decode_heads.seg_decode_head_with_time import FPN, SegformerDecodeHeadTemporal, SegformerDecodeHeadwTime
import warnings
import torch.nn.functional as F
import torch.nn as nn 
from transformers import AutoModel, AutoConfig
from transformers import SegformerFeatureExtractor, SegformerForImageClassification
import einops
import math
import torch
# from network.decode_heads import DeformableHeadWithTime

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(torch.expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = einops.rearrange(x, 'b -> b 1')
        freqs = x * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
     

class SegDiff(nn.Module):
    def __init__(self, 
                 bit_scale=0.01,
                 timesteps=3,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion="ddim",
                 accumulation=True,
                 num_classes=1) -> None:
        super().__init__()
        seg_model_path=r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512'
        self.encoder = SegformerForImageClassification.from_pretrained(seg_model_path)
        self.head_config = AutoConfig.from_pretrained(r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512')
        self.head_config.num_labels = num_classes
       # self.decoder_head = SegformerDecodeHead._from_config(self.head_config)
        
        self.bit_scale = bit_scale
        self.diffusion_type =diffusion
        self.num_classes = num_classes
        self.randsteps = randsteps
        self.accumulation = accumulation
        self.sample_range = sample_range
        self.timesteps = timesteps
        self.decode_head = SegformerDecodeHeadwTime._from_config(self.head_config)
        self.time_difference = time_difference
        self.x_inp_dim=64     
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.x_inp_dim)
        
        self.transform = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim*2),
            nn.Conv2d(self.x_inp_dim*2,self.x_inp_dim*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim*4,self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )
    
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        # time embeddings
        time_dim = 1024  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

    
    def extract_feat(self,img):
        _,_,H,W = img.size()
        output_feats = self.encoder(img, output_hidden_states=True)
        return list(output_feats.hidden_states)


    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, img):
        output_hidden_states = self.extract_feat(img)  # bs, 256, h/4, w/4
        x = output_hidden_states[0]
        b, c, h, w = x.size()
        device = x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = einops.repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            output_hidden_states[0] = feat
            
            mask_logit, bbox, clas= self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
            if self.accumulation:
                outs.append(mask_logit)
                # outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        ####TODO: Debug
        logit = resize(
            input=logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return logit, bbox, clas

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))


    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        if self.diffusion == "ddim":
            out, bbox, clas= self.ddim_sample(x, img_metas)
        # elif self.diffusion == 'ddpm':
        #     out = self.ddpm_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out, bbox, clas
    
    def forward(self, img, gt_semantic_seg=None, is_ddim=False):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Retu rns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if is_ddim:
            return self.ddim_sample(img)

        # backbone & neck
        output_hidden_states = self.extract_feat(img)  # bs, 256, h/4, w/4
        x = output_hidden_states[0]  # len=4
        batch, c, h, w = x.size()
        device = x.device
        
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes  # self.num_classes=1
        gt_down = self.embedding_table(gt_down.long()).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                    self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise

        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)  # embedding of img + noised mask Yt
        feat = self.transform(feat)
        output_hidden_states[0] = feat
        input_times = self.time_mlp(noise_level)
        mask_logit, bbox, clas= self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
        ####TODO: How to make sure the shape are the (H,W) instead of h/4, w/4?
        mask_logit = resize(
            input=mask_logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return mask_logit, bbox, clas
        #loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        #losses.update(loss_decode)
        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(
        #         [x], img_metas, gt_semantic_seg)
        #     losses.update(loss_aux)
        # return losses


    # def SegHeadforward(self, input):
    #     _,_,H,W = input.size()
    #     output_feats = self.encoder(input,output_hidden_states=True)
    #     outputs = self.decoder_head(output_feats.hidden_states)
    #     outputs_resized = resize(
    #         input=outputs,
    #         size=(H,W),
    #         mode='bilinear',
    #         align_corners=False)
    #     return outputs_resized
    
    
class DiffVPS(nn.Module):
    def __init__(self, 
                 bit_scale=0.01,
                 timesteps=3,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion="ddim",
                 accumulation=True,
                 num_classes=1) -> None:
        super().__init__()
        seg_model_path=r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512'
        self.encoder = SegformerForImageClassification.from_pretrained(seg_model_path)
        # Multitasks
        self.head_config = AutoConfig.from_pretrained(r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512')
        self.head_config.num_labels = num_classes
        self.decode_head = SegformerDecodeHeadwTime._from_config(self.head_config)
        # Reconstruction
        self.decoder_config = AutoConfig.from_pretrained(r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512')
        self.decoder_config.num_labels = 3
        self.temporal_encoder = SegformerForImageClassification.from_pretrained(seg_model_path)
        self.reconstruct_decoder = SegformerDecodeHead(config=self.decoder_config)
        # for p in self.encoder.parameters():
        #     p.requires_grad_(False) 
        
        self.with_neck = True
        self.neck = FPN(in_channel_list=[64, 128, 320, 512], out_channel=64)
        
        self.bit_scale = bit_scale
        self.diffusion_type =diffusion
        self.num_classes = num_classes
        self.randsteps = randsteps
        self.accumulation = accumulation
        self.sample_range = sample_range
        self.timesteps = timesteps
        
        self.time_difference = time_difference
        self.x_inp_dim=64     
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.x_inp_dim)

        self.transform = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim*2),
            nn.Conv2d(self.x_inp_dim*2,self.x_inp_dim*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim*4,self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        
        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        # time embeddings
        time_dim = 1024  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

    def extract_feat(self, img, seq_feats):
        # batch, num, c, height, width = seq.shape
        # seq = seq.reshape(batch * num, c, height, width)
        img_feats = self.encoder(img, output_hidden_states=True)
        img_feats = list(img_feats.hidden_states)
        # seq_feats = self.temporal_encoder(seq, output_hidden_states=True)
        # seq_feats = list(seq_feats.hidden_states)
        # Sequence level mean
        # for i in range(len(seq_feats)):
        #     _, c, h, w = seq_feats[i].shape
        #     seq_feats[i] = seq_feats[i].reshape(batch, num, c, h, w)
        #     seq_feats[i] = torch.mean(seq_feats[i], 1)  # (b, num, c, h, w) -> (b, c, h, w)
        # Multiscales
        for j in range(len(img_feats)):
            img_feats[j] = img_feats[j] + seq_feats[j]
        if self.with_neck == True:
            img_feats = self.neck(img_feats)
        return img_feats

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, img, seq_feats):
        output_hidden_states = self.extract_feat(img, seq_feats)  
        x = output_hidden_states[0]  # bs, 64, h/4, w/4
        b, c, h, w = x.size()
        device = x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = einops.repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            output_hidden_states[0] = feat
            
            mask_logit, bbox, clas = self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
            if self.accumulation:
                outs.append(mask_logit)
                # outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        ####TODO: Debug
        logit = resize(
            input=logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return logit

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def encode_decode(self, img, seq, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img, seq)
        if self.diffusion == "ddim":
            out, bbox, clas= self.ddim_sample(x, seq, img_metas)
        # elif self.diffusion == 'ddpm':
        #     out = self.ddpm_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out, bbox, clas
    
    def forward(self, img, seq, gt_semantic_seg=None, is_ddim=False):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Retu rns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # backbone & neck
        batch, num, c, height, width = seq.shape
        seq = seq.reshape(batch * num, c, height, width)
        seq_feats = self.temporal_encoder(seq, output_hidden_states=True)
        seq_feats = list(seq_feats.hidden_states)
        # Sequence level mean
        for i in range(len(seq_feats)):
            _, c, h, w = seq_feats[i].shape
            seq_feats[i] = seq_feats[i].reshape(batch, num, c, h, w)
            seq_feats[i] = torch.mean(seq_feats[i], 1)  # (b, num, c, h, w) -> (b, c, h, w)
            
        if is_ddim:
            return self.ddim_sample(img, seq_feats)

        output_hidden_states = self.extract_feat(img, seq_feats)  # bs, 256, h/4, w/4
        x = output_hidden_states[0]
        batch, c, h, w = x.size()
        device = x.device
        
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes  # self.num_classes=1
        gt_down = self.embedding_table(gt_down.long()).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                    self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise


        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)  # embedding of img + noised mask Yt
        feat = self.transform(feat)

        output_hidden_states[0] = feat

        input_times = self.time_mlp(noise_level)
        mask_logit, bbox, clas= self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
        mask_logit = resize(
            input=mask_logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        pred_frame = self.reconstruct_decoder(seq_feats)  # (4, batch, c, h, w)
        pred_frame = resize(
            input=pred_frame,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return mask_logit, bbox, clas, pred_frame
    
    
class Diff_Temporal(nn.Module):
    def __init__(self, 
                 bit_scale=0.01,
                 timesteps=3,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion="ddim",
                 accumulation=True,
                 num_classes=1) -> None:
        super().__init__()
        seg_model_path=r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512'
        self.encoder = SegformerForImageClassification.from_pretrained(seg_model_path)
        # Multitasks
        self.head_config = AutoConfig.from_pretrained(r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512')
        self.head_config.num_labels = num_classes
        self.decode_head = SegformerDecodeHeadTemporal._from_config(self.head_config)
        # Reconstruction
        self.decoder_config = AutoConfig.from_pretrained(r'./Diff_VPS/network/segformer-b3-finetuned-ade-512-512')
        self.decoder_config.num_labels = 3
        self.reconstruct_decoder = SegformerDecodeHead(config=self.decoder_config)
        # for p in self.encoder.parameters():
        #     p.requires_grad_(False) 
        self.bit_scale = bit_scale
        self.diffusion_type =diffusion
        self.num_classes = num_classes
        self.randsteps = randsteps
        self.accumulation = accumulation
        self.sample_range = sample_range
        self.timesteps = timesteps
        
        self.time_difference = time_difference
        self.x_inp_dim=64     
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.x_inp_dim)
        

        self.transform = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim*2),
            nn.Conv2d(self.x_inp_dim*2,self.x_inp_dim*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim*4,self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )
    
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')
        
        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        # time embeddings
        time_dim = 1024  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

    
    def extract_feat(self, img, seq):
        batch, num, c, height, width = seq.shape
        seq = seq.reshape(batch * num, c, height, width)
        img_feats = self.encoder(img, output_hidden_states=True)
        img_feats = list(img_feats.hidden_states)
        seq_feats = self.encoder(seq, output_hidden_states=True)
        seq_feats = list(seq_feats.hidden_states)
        # Sequence level mean
        for i in range(len(seq_feats)):
            _, c, h, w = seq_feats[i].shape
            seq_feats[i] = seq_feats[i].reshape(batch, num, c, h, w)
            seq_feats[i] = torch.mean(seq_feats[i], 1)  # (b, num, c, h, w) -> (b, c, h, w)
        # Multiscales
        for j in range(len(img_feats)):
            img_feats[j] = img_feats[j] + seq_feats[j]
        return img_feats

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, img, seq):
        output_hidden_states = self.extract_feat(img, seq)  
        x = output_hidden_states[0]  # bs, 64, h/4, w/4
        b, c, h, w = x.size()
        device = x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = einops.repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            output_hidden_states[0] = feat
            
            mask_logit = self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
            if self.accumulation:
                outs.append(mask_logit)
                # outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        ####TODO: Debug
        logit = resize(
            input=logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return logit

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))
    
    def forward(self, img, seq, gt_semantic_seg=None, is_ddim=False):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Retu rns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if is_ddim:
            return self.ddim_sample(img, seq)

        # backbone & neck
        output_hidden_states = self.extract_feat(img, seq)  # bs, 256, h/4, w/4
        x = output_hidden_states[0]  # len=4
        batch, c, h, w = x.size()
        device = x.device
        
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes  # self.num_classes=1
        gt_down = self.embedding_table(gt_down.long()).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                    self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise

        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)  # embedding of img + noised mask Yt
        feat = self.transform(feat)
        output_hidden_states[0] = feat
        input_times = self.time_mlp(noise_level)
        mask_logit = self.decode_head(output_hidden_states, input_times)  # [bs, 150, ]
        mask_logit = resize(
            input=mask_logit,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        pred_frame = self.reconstruct_decoder(output_hidden_states)  # (4, batch, c, h, w)
        pred_frame = resize(
            input=pred_frame,
            size=(4*h, 4*w),
            mode='bilinear',
            align_corners=False)
        
        return mask_logit, pred_frame