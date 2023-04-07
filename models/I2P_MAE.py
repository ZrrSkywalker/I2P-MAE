import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
import random
from extensions.chamfer_dist import ChamferDistanceL2
from torch_scatter import scatter

from utils.logger import *
from .modules import *
import clip_attn as clip


''' Hierarchical Encoder with 2D-guided Masking '''
class H_Encoder(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.mask_ratio = config.mask_ratio 
        self.encoder_depths = config.encoder_depths
        self.encoder_dims =  config.encoder_dims
        self.local_radius = config.local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=config.num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def rand_mask(self, center):
        B, G, _ = center.shape
        self.num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device) # B G

    ''' 2D-guided Masking '''
    def mask_2D_guided(self, center, spatial_sals=None):
        B, G, _ = center.shape
        
        attm = F.softmax(spatial_sals, dim=1)  
        ids_shuffle = torch.multinomial(attm, G, replacement=False)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        self.num_mask = int(self.mask_ratio * G)
        self.num_keep = G - self.num_mask

        overall_mask = torch.ones([B, G], device=center.device)
        overall_mask[:, 0: self.num_keep] = 0
        
        overall_mask = torch.gather(overall_mask, dim=1, index=ids_restore).to(torch.bool)
        return overall_mask  # B G

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False, spatial_sals=None):
        # generate mask at the highest level
        bool_masked_pos = []
        if eval:
            # no mask
            B, G, _ = centers[-1].shape
            bool_masked_pos.append(torch.zeros(B, G).bool().cuda())
        else:
            # mask_index: 1, mask; 0, vis
            bool_masked_pos.append(self.mask_2D_guided(centers[-1], spatial_sals=spatial_sals))

        # Multi-scale Masking by back-propagation
        for i in range(len(neighborhoods) - 1, 0, -1):
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)
            idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx
            idx_masked = idx_masked.reshape(-1).long()
            masked_pos = torch.ones(b * centers[i - 1].shape[1]).cuda().scatter(0, idx_masked, 0).bool()
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))

        # hierarchical encoding
        bool_masked_pos.reverse()
        x_vis_list = []
        mask_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            # visible_index
            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            # Due to Multi-scale Masking different, samples of a batch have varying numbers of visible tokens
            # find the longest visible sequence in the batch
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = torch.max(vis_tokens_len)
            # use the longest length (max_tokens_len) to construct tensors
            x_vis = torch.zeros(batch_size, max_tokens_len, C).cuda()
            masked_center = torch.zeros(batch_size, max_tokens_len, 3).cuda()
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len).cuda()
            
            for bz in range(batch_size):
                # inject valid visible tokens
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                x_vis[bz][0: vis_tokens_len[bz]] = vis_tokens
                # inject valid visible centers
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0: vis_tokens_len[bz]] = vis_centers
                # the mask for valid visible tokens/centers
                mask_vis[bz][0: vis_tokens_len[bz], 0: vis_tokens_len[bz]] = 0
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                # disabled for pre-training, this step would not change mask_vis by *
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.encoder_pos_embeds[i](masked_center)

            x_vis = self.encoder_blocks[i](x_vis, pos, mask_vis_att)
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))

            if i == len(centers) - 1:
                pass
            else:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]
                x_vis = group_input_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos


@MODELS.register_module()
class I2P_MAE(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[I2P-MAE]', logger ='I2P-MAE')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group_v2(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                        embed_dim=self.decoder_dims[i],
                        depth=self.decoder_depths[i],
                        drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                        num_heads=config.num_heads,
                    ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                        nn.Linear(3, self.decoder_dims[i]),
                        nn.GELU(),
                        nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
                    ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                                self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                                blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                            ))  
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # 3D-coordinate reconstruction                        
        self.rec_head_3d = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[1], 1)
        self.loss_3d = ChamferDistanceL2().cuda()
        
        # 2D-semantic reconstruction
        self.feat_dim_2d = config.clip_config.feat_dim
        self.rec_head_2d = nn.Conv1d(self.decoder_dims[-1], self.feat_dim_2d * self.group_sizes[1] * 3, 1)
        self.loss_2d = nn.MSELoss()

        # multi-view projection
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])
        self.img_size = 224
        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], 
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        self.proj_reduction = 'sum'

        # 2D pre-trained models, clip by default
        self.clip_model, _ = clip.load(config.clip_config.visual_encoder)
        self.clip_model.eval()

    ''' Efficient Projection '''
    def proj2img(self, pc):
        B, N, _ = pc.shape
        
        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.img_size - 3)  # B,

        # Point Index
        pc_min = pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # Point Densify
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.img_offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(idx_xy.size(0), N*25, 2) + 1
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        
        # Object to Image Center
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.img_size / 2 - idx_xy_dense_center[:, 0: 1] - 1
        offset_y = self.img_size / 2 - idx_xy_dense_center[:, 1: 2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Expand Point Features
        f_dense = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3].repeat(1, 1, 3)
        
        idx_zero = idx_xy_dense_offset < 0
        idx_obj = idx_xy_dense_offset > 223
        idx_xy_dense_offset = idx_xy_dense_offset + idx_zero.to(torch.int32)
        idx_xy_dense_offset = idx_xy_dense_offset - idx_obj.to(torch.int32)

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size-1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]
        
        # Get Image Features
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce=self.proj_reduction) 

        # need to pad 
        if out.size(1) < self.img_size * self.img_size: 
            delta = self.img_size * self.img_size - out.size(1) 
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device) 
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        else: 
            res = out.reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        
        # B 224 224 C
        img = res.permute(0, 3, 1, 2).contiguous()
        mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()  # 1 3 1 1
        std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()   # 1 3 1 1
        # Normalize the pic        
        img = nn.Sigmoid()(img)
        img_norm = img.sub(mean_vec).div(std_vec)
        return img_norm, pc_min, grid_size, (offset_x, offset_y)

    ''' Image-to-Point Back-projection '''
    def I2P(self, pc, f, pc_min, grid_size, offsets):
        B, N, _ = pc.shape

        # Point Index
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # Point Densify
        idx_xy_dense = idx_xy + 1
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        
        # Object to Image Center
        idx_xy_dense_offset = idx_xy_dense + torch.cat(offsets, dim=1).unsqueeze(dim=1)  # B, N, 2

        # Expand Point Features
        B, C, H, W = f.shape
        f_dense = F.interpolate(f, size=(self.img_size, self.img_size), mode='bicubic').reshape(B, C, -1).permute(0, 2, 1)  # B, N, C

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size - 1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]  # B, N
        
        # Get Image Features
        out = torch.gather(f_dense, 1, new_idx_xy_dense.to(dtype=torch.int64).unsqueeze(-1).repeat(1, 1, C))
        return out      

    def forward(self, pts, eval=False, **kwargs):
        # multi-scale representations of point clouds
        neighborhoods, centers, idxs, neighborhood_oris = [], [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx, neighborhood_ori = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx, neighborhood_ori = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            # original 3D coordinates without normalization by centers
            neighborhood_oris.append(neighborhood_ori)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        if eval:
            # for linear svm
            x_vis_list, mask_vis_list, _ = self.h_encoder(neighborhoods, centers, idxs, eval=True)
            x_vis = x_vis_list[-1]
            return x_vis.max(1)[0]

        else:
            ''' Project Point Clouds along 3 Axis '''
            pts_1 = pts
            pts_2 = torch.cat((pts[..., 2: 3], pts[..., 0: 2]), dim=-1)
            pts_3 = torch.cat((pts[..., 1: 3], pts[..., 0: 1]), dim=-1)

            imgs_1, pc_min_1, grid_size_1, offsets_1 = self.proj2img(pts_1)  # b, 3, 224, 224
            imgs_2, pc_min_2, grid_size_2, offsets_2 = self.proj2img(pts_2)  # b, 3, 224, 224
            imgs_3, pc_min_3, grid_size_3, offsets_3 = self.proj2img(pts_3)  # b, 3, 224, 224

            with torch.no_grad():
                
                ''' 2D Representations Extraction '''
                imgs = torch.cat((imgs_1, imgs_2, imgs_3), dim=0)
                img_feats, img_sals = self.clip_model.encode_image(imgs)  # b, c, h, w
                img_feats = img_feats.float()
                img_sals = img_sals.float()[:, 0, 1:]  # b, 196
                
                ''' 2D Visual Features '''
                B, _, _, _ = img_feats.shape
                img_feats_1 = img_feats[0: B // 3]
                img_feats_2 = img_feats[B // 3: B // 3 * 2]
                img_feats_3 = img_feats[B // 3 * 2: ]
                
                ''' 2D Attention Maps '''
                img_sals = img_sals.reshape(-1, 1, 14, 14)  # b, 1, 14, 14
                img_sals_1 = img_sals[0: B // 3]
                img_sals_2 = img_sals[B // 3: B // 3 * 2]
                img_sals_3 = img_sals[B // 3 * 2: ]

                '''' Spatial Attention Cloud '''            
                pts_1 = centers[-1]
                pts_2 = torch.cat((pts_1[..., 2: 3], pts_1[..., 0: 2]), dim=-1)
                pts_3 = torch.cat((pts_1[..., 1: 3], pts_1[..., 0: 1]), dim=-1)
                
                spatial_sals_1 = self.I2P(pts_1, img_sals_1, pc_min_1, grid_size_1, offsets_1).squeeze()  # b, center_num
                spatial_sals_2 = self.I2P(pts_2, img_sals_2, pc_min_2, grid_size_2, offsets_2).squeeze()  # b, center_num
                spatial_sals_3 = self.I2P(pts_3, img_sals_3, pc_min_3, grid_size_3, offsets_3).squeeze()  # b, center_num
                spatial_sals = (spatial_sals_1 + spatial_sals_2 + spatial_sals_3) / 3

            x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs, spatial_sals=spatial_sals)

        # hierarchical decoder
        centers.reverse()
        neighborhoods.reverse()
        neighborhood_oris.reverse()
        x_vis_list.reverse()
        masks.reverse()

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i]
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_full, mask_token], dim=1)
            
            else:
                x_vis = x_vis_list[i]
                bool_vis_pos = ~masks[i]
                mask_vis = mask_vis_list[i]
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis]

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full)
            
            
        x_full  = self.decoder_norm(x_full)
        B, N, C = x_full.shape

        ''' 3D-coordinate Reconstruction '''
        x_rec_mask = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec_mask.shape

        rec_3d = self.rec_head_3d(x_rec_mask.unsqueeze(-1)).reshape(L, -1, 3)
        gt_3d = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)
        loss_3d = self.loss_3d(rec_3d, gt_3d)

        ''' 2D-semantic Reconstruction '''
        x_rec_vis = x_full[~masks[-2]].reshape(-1, C)
        L, _ = x_rec_vis.shape

        rec_2d = self.rec_head_2d(x_rec_vis.unsqueeze(-1)).reshape(L, -1, self.feat_dim_2d * 3)
        
        # 2D-semantic targets
        rebuild_xyz = neighborhood_oris[-2]
        B, N, K, _ = rebuild_xyz.shape
        rebuild_xyz = rebuild_xyz.reshape(B, N * K, 3)
        
        rebuild_xyz_1 = rebuild_xyz
        rebuild_xyz_2 = torch.cat((rebuild_xyz[..., 2: 3], rebuild_xyz[..., 0: 2]), dim=-1)
        rebuild_xyz_3 = torch.cat((rebuild_xyz[..., 1: 3], rebuild_xyz[..., 0: 1]), dim=-1)
        
        clip_feats_1 = self.I2P(rebuild_xyz_1, img_feats_1, pc_min_1, grid_size_1, offsets_1)
        clip_feats_1 = clip_feats_1.reshape(B, N, K, -1)[~masks[-2]].reshape(L, -1, self.feat_dim_2d)

        clip_feats_2 = self.I2P(rebuild_xyz_2, img_feats_2, pc_min_2, grid_size_2, offsets_2)
        clip_feats_2 = clip_feats_2.reshape(B, N, K, -1)[~masks[-2]].reshape(L, -1, self.feat_dim_2d)

        clip_feats_3 = self.I2P(rebuild_xyz_3, img_feats_3, pc_min_3, grid_size_3, offsets_3)
        clip_feats_3 = clip_feats_3.reshape(B, N, K, -1)[~masks[-2]].reshape(L, -1, self.feat_dim_2d)

        loss_2d = self.loss_2d(rec_2d, torch.cat((clip_feats_1, clip_feats_2, clip_feats_3), dim=-1))
        
        return (loss_3d, loss_2d)
