import torch.nn as nn
from transformers import SegformerConfig,SegformerPreTrainedModel
import torch
import math
import einops
import torch.nn.functional as F


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(FPN, self).__init__()
        inner_layer = []
        out_layer = []
        for in_channel in in_channel_list:
            inner_layer.append(nn.Conv2d(in_channel, out_channel, 1))
            out_layer.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        # self.upsample=nn.Upsample(size=, mode='nearest')
        self.inner_layer = nn.ModuleList(inner_layer)
        self.out_layer = nn.ModuleList(out_layer)
        
    def forward(self, x):
        head_output = []
        corent_inner = self.inner_layer[-1](x[-1])
        head_output.append(self.out_layer[-1](corent_inner))
        for i in range(len(x) - 2, -1, -1):
            pre_inner = corent_inner
            corent_inner = self.inner_layer[i](x[i])
            size = corent_inner.shape[2:]
            pre_top_down = F.interpolate(pre_inner, size=size)
            add_pre2corent = pre_top_down + corent_inner
            head_output.append(self.out_layer[i](add_pre2corent))
        return list(reversed(head_output))


class SegformerDecodeHeadwTime(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        time_mlps = []
        self.time_embed_dim = 1024
        self.in_channels = [64, 128, 320, 512]
        for i in range(config.num_encoder_blocks):
            # mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])  # [64, 128, 320, 512]
            mlp = SegformerMLP(config, input_dim=64)  # [64, 64, 64, 64]
            mlps.append(mlp)
            time_mlp = nn.Sequential( # [2, 1024]
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, 768*2) # [2, 512]
            ) 
            time_mlps.append(time_mlp)
            
        self.linear_c = nn.ModuleList(mlps)
        
        self.mlp_time_c = nn.ModuleList(time_mlps)
        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.flat = nn.Flatten()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear_clas = nn.Linear(768, 6)  # nn.Linear(512, 6)
        
        # self.fpn = FPN(in_channel_list=[64, 128, 320, 512], out_channel=256)
        self.bbox_fuse = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=1,
            bias=False,
        )
        self.linear_b_0 = nn.Linear(64, 16)  # nn.Linear(512, 64) channels*height*width 
        self.linear_b_1 = nn.Linear(16, 4)

        self.config = config
        

    def forward(self, encoder_hidden_states, time) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        all_bbox_states = ()
        bbox_states = []
        for encoder_hidden_state, mlp, time_mlp in zip(encoder_hidden_states, self.linear_c, self.mlp_time_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )
            bbox_states.append(encoder_hidden_state)
            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            
           
            time_embed = time_mlp(time)  
            time_embed = einops.rearrange(time_embed, 'b c -> b c 1 1') 
            scale1, shift1 = time_embed.chunk(2, dim=1)  
            
            encoder_hidden_state = encoder_hidden_state * (scale1+1) + shift1
            
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )

            all_hidden_states += (encoder_hidden_state,)
        
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        
        _clas = self.pool(hidden_states)  # torch.Size([b, 768, 1, 1])
        _clas = self.flat(_clas)  # torch.Size([b, 768])
        clas = self.linear_clas(_clas)  # torch.Size([6])
        
        # bbox_states = self.fpn(bbox_states)
        for i in range(len(bbox_states)):
            bbox_states[i] = nn.functional.interpolate(bbox_states[i], bbox_states[0].size()[2:], mode="bilinear", align_corners=False)
            all_bbox_states += (bbox_states[i],)  
        _bbox_states = self.bbox_fuse(torch.cat(all_bbox_states[::-1], dim=1))
        _bbox_states = self.pool(_bbox_states)
        _bbox_states = self.flat(_bbox_states)
        _bbox_states = F.relu(self.linear_b_0(_bbox_states))
        bbox = F.relu(self.linear_b_1(_bbox_states))
        
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits, bbox, clas
    
    
class SegformerDecodeHeadTemporal(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        time_mlps = []
        self.time_embed_dim = 1024
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
            time_mlp = nn.Sequential( # [2, 1024]
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, 768*2) # [2, 512]
            ) 
            time_mlps.append(time_mlp)
            
        self.linear_c = nn.ModuleList(mlps)
        
        self.mlp_time_c = nn.ModuleList(time_mlps)
        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config
        

    def forward(self, encoder_hidden_states, time) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        all_bbox_states = ()
        bbox_states = []
        for encoder_hidden_state, mlp, time_mlp in zip(encoder_hidden_states, self.linear_c, self.mlp_time_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )
            bbox_states.append(encoder_hidden_state)
            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            
           
            time_embed = time_mlp(time)  
            time_embed = einops.rearrange(time_embed, 'b c -> b c 1 1') 
            scale1, shift1 = time_embed.chunk(2, dim=1)  
            
            encoder_hidden_state = encoder_hidden_state * (scale1+1) + shift1
            
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )

            all_hidden_states += (encoder_hidden_state,)
        
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits
