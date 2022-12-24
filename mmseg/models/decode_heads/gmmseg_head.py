# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from mmseg.utils.distributed import concat_all_gather_wo_grad
from mmseg.utils.distributions import MultivariateNormalDiag
from mmseg.utils.GMMSeg import distributed_sinkhorn_wograd, shifted_var, init_weights, l2_normalize, momentum_update, rnd_sample


class GMMSegHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        if self.channels != embedding_dim:
            self.projection = ConvModule(
                in_channels=self.channels,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=self.norm_cfg)
        else: self.projection = None
        self.embedding_dim = embedding_dim
        self.num_components = decoder_params['num_components']
        self.update_GMM_interval = decoder_params['update_GMM_interval']

        gamma = decoder_params['gamma']
        self.gamma_mean = gamma if isinstance(decoder_params['gamma'],(float,int)) else gamma[0]
        self.gamma_cov = gamma if isinstance(decoder_params['gamma'],(float,int)) else gamma[1]
        self.factors = [decoder_params['factor_n'], decoder_params['factor_c'], decoder_params['factor_p']]

        self.K = decoder_params['mem_size']
        self.Ks = torch.tensor([self.K for _c in range(self.num_classes*self.num_components)], dtype=torch.long)
        
        self.max_sample_size = decoder_params['max_sample_size']
        self.register_buffer("queue", torch.randn(self.num_classes*self.num_components, embedding_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=-2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes*self.num_components, dtype=torch.long))

        self.apply(init_weights)

        self.means = nn.Parameter(torch.zeros(self.num_classes, self.num_components, embedding_dim), requires_grad=False)
        trunc_normal_(self.means, std=0.02)
        self.num_prob_n = self.num_components
        self.diagonal = nn.Parameter(torch.ones(self.num_classes,self.num_components,self.embedding_dim), requires_grad=False)
        self.eye_matrix = nn.Parameter(torch.ones(embedding_dim), requires_grad=False)
        self.feat_norm = nn.LayerNorm(embedding_dim) 
        self.mask_norm = nn.LayerNorm(self.num_classes) 
        
        # 
        self.iteration_counter = nn.Parameter(torch.zeros(1), requires_grad=False)


    @abstractmethod
    def base_feature_transform(self, inputs):
        pass


    def forward(self, inputs, gt_semantic_seg=None, train_cfg=None, test_cfg=None):
        base_feature = self.base_feature_transform(inputs)

        _c = rearrange(base_feature, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c) # * n, d
        _c = l2_normalize(_c)

        self.means.data.copy_(l2_normalize(self.means))

        _log_prob = self.compute_log_prob(_c)
        final_probs = _log_prob.contiguous().view(-1, self.num_classes, self.num_prob_n)

        _m_prob = torch.amax(final_probs, dim=-1)

        out_seg = self.mask_norm(_m_prob)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=base_feature.shape[0], h=base_feature.shape[2])

        if gt_semantic_seg is not None: 
            # resize groundtruth to fit feature size
            gt_seg_full = resize(gt_semantic_seg.float(), size=base_feature.size()[2:], mode='nearest')
            gt_seg = gt_seg_full.view(-1)

            contrast_logits, contrast_target, qs = self.online_contrast(gt_seg, final_probs, _c, out_seg)

            with torch.no_grad():
                # * update memory
                _c_mem = concat_all_gather_wo_grad(_c)
                _gt_seg_mem = concat_all_gather_wo_grad(gt_seg)
                _qs = concat_all_gather_wo_grad(qs)

                unique_c_list = _gt_seg_mem.unique().int()
                for k in unique_c_list:
                    if k == 255: continue
                    self._dequeue_and_enqueue_k(k.item(), _c_mem, _qs.bool(), (_gt_seg_mem == k.item()))

                # * EM
                if self.iteration_counter % self.update_GMM_interval == 0:
                    self.update_GMM(unique_c_list)

            return out_seg, contrast_logits, contrast_target

        return out_seg
    

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, contrast_logits, contrast_target = self.forward(inputs, gt_semantic_seg=gt_semantic_seg, train_cfg=train_cfg)

        losses = self.losses(seg_logits, gt_semantic_seg)

        if train_cfg['contrast_loss'] is True and contrast_logits is not None:
            loss_contrast = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=255)
            losses['loss_contrast'] = loss_contrast * train_cfg['contrast_loss_weight']

        self.iteration_counter += 1

        return losses 


    def compute_log_prob(self, _fea):
        covariances = self.diagonal.detach_() # * c,p,d,d

        _prob_n = []
        _n_group = _fea.shape[0] // self.factors[0]
        _c_group = self.num_classes // self.factors[1]
        for _c in range(0,self.num_classes,_c_group):
            _prob_c = []
            _c_means = self.means[_c:_c+_c_group]
            _c_covariances = covariances[_c:_c+_c_group]

            _c_gauss = MultivariateNormalDiag(_c_means.view(-1, self.embedding_dim), scale_diag=_c_covariances.view(-1,self.embedding_dim)) # * c*p multivariate gaussian
            for _n in range(0,_fea.shape[0],_n_group):
                _prob_c.append(_c_gauss.log_prob(_fea[_n:_n+_n_group,None,...]))
            _c_probs = torch.cat(_prob_c, dim=0) # n, cp
            _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], -1, self.num_prob_n)
            _prob_n.append(_c_probs)
        probs = torch.cat(_prob_n, dim=1)

        return probs.contiguous().view(probs.shape[0],-1)

        
    @torch.no_grad()
    def _dequeue_and_enqueue_k(self, _c, _c_embs, _c_cluster_q, _c_mask):

        if _c_mask is None: _c_mask = torch.ones(_c_embs.shape[0]).detach_()

        _k_max_sample_size = self.max_sample_size
        _embs = _c_embs[_c_mask>0]
        _cluster = _c_cluster_q[_c_mask>0]
        for q_index in range(self.num_components):
            _q_ptr = _c*self.num_components+q_index
            ptr = int(self.queue_ptr[_q_ptr])
            
            if torch.sum(_cluster[:, q_index]) == 0: continue
            assert _cluster[:, q_index].shape[0] == _embs.shape[0]
            _q_embs = _embs[_cluster[:, q_index]]

            _q_sample_size = _q_embs.shape[0]
            assert _q_sample_size == torch.sum(_cluster[:, q_index])

            if self.max_sample_size != -1 and _q_sample_size > _k_max_sample_size:
                _rnd_sample = rnd_sample(_q_sample_size, _k_max_sample_size, _uniform=True, _device=_c_embs.device)
                _q_embs = _q_embs[_rnd_sample, ...]
                _q_sample_size = _k_max_sample_size

            # replace the embs at ptr (dequeue and enqueue)
            if ptr + _q_sample_size >= self.Ks[_q_ptr]:
                _fir = self.Ks[_q_ptr] - ptr
                _sec = _q_sample_size - self.Ks[_q_ptr] + ptr
                self.queue[_q_ptr, :, ptr:self.Ks[_q_ptr]] = _q_embs[:_fir].T
                self.queue[_q_ptr, :, :_sec] = _q_embs[_fir:].T
            else:
                self.queue[_q_ptr, :, ptr:ptr + _q_sample_size] = _q_embs.T
            
            ptr = (ptr + _q_sample_size) % self.Ks[_q_ptr]  # move pointer
            self.queue_ptr[_q_ptr] = ptr

    
    @torch.no_grad()
    def update_GMM(self, unique_c_list):
        components = self.means.data.clone() 
        covs = self.diagonal.data.clone()

        for _c in unique_c_list:
            if _c == 255: continue
            _c = _c if isinstance(_c, int) else _c.item()

            for _p in range(self.num_components):
                _p_ptr = _c*self.num_components + _p
                _mem_fea_q = self.queue[_p_ptr,:,:self.Ks[_c]].transpose(-1,-2) # n,d

                f = l2_normalize(torch.sum(_mem_fea_q, dim=0)) # d,

                new_value = momentum_update(old_value=components[_c, _p, ...], new_value=f, momentum=self.gamma_mean, debug=False)
                components[_c, _p, ...] = new_value

                _shift_fea = _mem_fea_q - f[None, ...] # * n, d

                _cov = shifted_var(_shift_fea, rowvar=False)
                _cov = _cov + 1e-2 * self.eye_matrix
                _cov = _cov.sqrt()

                new_covariance = momentum_update(old_value=covs[_c, _p, ...], new_value=_cov, momentum=self.gamma_cov, debug=False)
                covs[_c, _p, ...] = new_covariance
        
        self.means = nn.Parameter(components, requires_grad=False)
        self.diagonal = nn.Parameter(covs, requires_grad=False)
        # * NOTE: need not to sync across gpus. memory is shared across all gpus


    def online_contrast(self, gt_seg, simi_logits, _c, out_seg):
        # find pixels that are correctly classified
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        # compute logits
        contrast_logits = simi_logits.flatten(1) # * n, c*p
        contrast_target = gt_seg.clone().float()

        return_qs = torch.zeros(size=(simi_logits.shape[0], self.num_components), device=gt_seg.device)
        # clustering for each class
        for k in gt_seg.unique().long():
            if k == 255: continue
            # get initial assignments for the k-th class
            init_q = simi_logits[:, k, :]
            init_q = init_q[gt_seg == k, ...] # n,p
            init_q = init_q[:,:self.num_components]
            init_q = init_q / torch.abs(init_q).max()

            # * init_q: [gt_n, p]
            # clustering q.shape = n x self.num_components
            q, indexs = distributed_sinkhorn_wograd(init_q)
            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                # * process nan
                q[torch.isnan(q)] = 0
                indexs[torch.isnan(q).int().sum(dim=1)>0] = 255 - (self.num_prob_n * k)

            # binary mask for pixels of the k-th class
            m_k = mask[gt_seg == k]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_components)
            # mask the incorrect q with zero
            q = q * m_k_tile  # n x self.num_prob_n

            contrast_target[gt_seg == k] = indexs.float() + (self.num_prob_n * k)

            return_qs[gt_seg == k] = q

        return contrast_logits, contrast_target, return_qs
