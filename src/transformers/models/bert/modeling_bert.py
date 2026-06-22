# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""
Totalsum=0
TotalCounts=0

MSBFirstround=0
import math
import time
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert import HyperParameters
from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, auto_docstring, get_torch_version, logging
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        #change #1
        #Convert the embedings to FIxed point INT 16
        
        embeddings=torch.round(embeddings*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        embeddings=torch.clip(embeddings,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    def compute_importance_l1(self, key_layer):
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        """L1 norm across feature dimension"""
        # print("key_layer")
        # print(key_layer)
        # key_layer: [B, 12, n, 64]
        # x1 = key_layer.unsqueeze(3)         # [B, 12, n, 1, 64]
        # x2 = key_layer.unsqueeze(2)         # [B, 12, 1, n, 64]
        # l1 = torch.sum(torch.abs(x1 - x2), dim=-1)  # [B, 12, n, n]
        #another way to implemant the same code
        B, H, n, d = key_layer.shape
        # Reshape to [B*H, n, d] for cdist, then reshape back
        key_flat = key_layer.reshape(B * H, n, d)
        l1 = torch.cdist(key_flat, key_flat, p=1)   # [B*H, n, n]
        l1 = l1.reshape(B, H, n, n)
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'Compute_importance_LI' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return l1
    def compute_importance_Normalized_l1(self, key_layer):
        # Normalize along feature dimension (L1 normalization)
        key_layer = key_layer / (torch.sum(torch.abs(key_layer), dim=-1, keepdim=True) + 1e-8)
        """L1 norm across feature dimension"""
        # key_layer: [B, 12, n, 64]
        x1 = key_layer.unsqueeze(3)         # [B, 12, n, 1, 64]
        x2 = key_layer.unsqueeze(2)         # [B, 12, 1, n, 64]
        l1 = torch.sum(torch.abs(x1 - x2), dim=-1)  # [B, 12, n, n]
        return l1
    def compute_importance_l2(self, key_layer):
        # key_layer: [B, 12, n, 64]
        x1 = key_layer.unsqueeze(3)         # [B, 12, n, 1, 64]
        x2 = key_layer.unsqueeze(2)         # [B, 12, 1, n, 64]
        l2 = torch.norm(x1 - x2, dim=-1)    # [B, 12, n, n]
        return l2
    
    def hard_leader_from_distance(self,dist,tau,keep_diag: bool = True):
        """
        hard-Leader clustering driven directly by an L1-distance matrix.

        Parameters
        ----------
        dist : Tensor  [n, n]
            Pair-wise (symmetric) distance matrix.
        tau  : float
            Threshold – two tokens are considered “close” if dist <= tau.
        keep_diag : bool
            • If True  (default) we force the diagonal to be `True`
              so every token is always in its own row-cluster.
            • If False, self-edges are treated the same as any other entry.

        Returns
        -------
        clusters : list[list[int]]
            Greedy (order-dependent) clusters.
            A token appears in the first cluster whose leader row reaches it.
        """
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
       #1.  Build mask  (True ⇔ distance ≤ tau)
        # print("dist.shape",dist.shape)
        mask = (dist <= tau)
        # print("mask",mask)
        
        B, H, n, _ = mask.shape
        clusters_by_head = []                      # <- final nested list

        for b in range(B):                         # outer loop over batches
            batch_list = []                        # will collect H heads
            for h in range(H):                     # 12 heads in ViT/BERT
                slice_mask = mask[b, h]            # [n, n] for this head
                visited    = torch.zeros(n, dtype=torch.bool, device=mask.device)
                clusters   = []                    # clusters for this head
                # Always cluster CLS (first token) alone
                clusters.append([0])
                visited[0] = True
                
                # ---- greedy row scan ----
                for i in range(n):
                    if visited[i]:
                        continue                   # token i already placed
                    members = torch.where(slice_mask[i]& ~visited)[0]
                    # Remove 0 and n-1 from members if present
                    members = members[(members != 0) & (members != n-1)]
                    if len(members) > 0:
                        clusters.append(members.tolist())
                        visited[members] = True

                # Always cluster SEP (last token) alone
                clusters.append([n-1])
                visited[n-1] = True

                batch_list.append(clusters)        # one head done
            clusters_by_head.append(batch_list)    # one batch done
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'hard_leader_from_distance' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return clusters_by_head
    
    def hard_leader_from_distance_tensor(self,dist, tau, keep_diag: bool = True):
        """
        Perform hard-leader clustering based on L1 distance matrix.

        Args:
            dist: Tensor of shape [B, H, n, n] (batch, head, tokens, tokens)
            tau: float threshold — connect tokens with distance ≤ tau
            keep_diag: if True, keep self-connections for all tokens (not used in this code)

        Returns:
            clusters_tensor: Tensor of shape [B, H, max_num_clusters, max_cluster_size],
                             where padded values are -1
        """
        torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        start = time.time()
        mask = (dist <= tau)  # shape: [B, H, n, n]
        B, H, n, _ = mask.shape

        all_clusters = [[[] for _ in range(H)] for _ in range(B)]
        max_num_clusters = 0
        max_cluster_size = 0

        for b in range(B):
            for h in range(H):
                slice_mask = mask[b, h]  # shape: [n, n]
                visited = torch.zeros(n, dtype=torch.bool, device=mask.device)
                clusters = []

                # Cluster CLS token alone
                clusters.append([0])
                visited[0] = True

                for i in range(n):
                    if visited[i]:
                        continue
                    members = torch.where(slice_mask[i] & ~visited)[0]
                    members = members[(members != 0) & (members != n - 1)]
                    if len(members) > 0:
                        clusters.append(members.tolist())
                        visited[members] = True

                # Cluster SEP token alone
                clusters.append([n - 1])
                visited[n - 1] = True

                all_clusters[b][h] = clusters
                max_num_clusters = max(max_num_clusters, len(clusters))
                max_cluster_size = max(
                    max_cluster_size,
                    max(len(g) for g in clusters) if clusters else 0
                )

        # Create padded tensor with -1
        clusters_tensor = torch.full(
            (B, H, max_num_clusters, max_cluster_size),
            fill_value=-1,
            dtype=torch.long,
            device=dist.device
        )

        for b in range(B):
            for h in range(H):
                for i, group in enumerate(all_clusters[b][h]):
                    clusters_tensor[b, h, i, :len(group)] = torch.tensor(group, device=dist.device)
        torch.cuda.synchronize()  # Wait for matmul to finish
        end = time.time() 
        print("time in 'hard_leader_from_distance_tensor' Fun =")
        print(f"Time on GPU: {end - start:.6f} seconds")
        return clusters_tensor
    
    
    def union_find_labels(self,mask: torch.Tensor):
        N = mask.size(0)
        parent = torch.arange(N, device=mask.device)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(N):
            nbrs = torch.where(mask[i])[0]
            for j in nbrs:
                ri = find(i)
                rj = find(j.item())
                if ri != rj:
                    parent[ri] = rj
        for i in range(N):
            parent[i] = find(i)
        return parent

    def cluster_one_graph(self,dist_slice: torch.Tensor, tau: float):
        N = dist_slice.size(0)
        mask = dist_slice <= tau
        mask.fill_diagonal_(False)
        mask[0, :] = mask[:, 0] = False
        mask[N-1, :] = mask[:, N-1] = False
        labels = self.union_find_labels(mask)
        clusters = [[0]]
        added = set([0, N-1])
        for i in range(1, N-1):
            if labels[i].item() == i and (mask[i].any()):
                members = torch.where(labels == i)[0]
                clusters.append(members.tolist())
                added.update(members.tolist())
        clusters.append([N-1])
        maxC = len(clusters)
        maxS = max(len(c) for c in clusters)
        out = dist_slice.new_full((maxC, maxS), -1, dtype=torch.long)
        for ci, grp in enumerate(clusters):
            out[ci, :len(grp)] = torch.tensor(grp, device=dist_slice.device)
        return out

    def hard_leader_batched(self,dist: torch.Tensor, tau: float):
        """
        Batched clustering over [B, H, N, N], fully on GPU, but uses a for-loop over B and H.
        """
        # torch.cuda.synchronize()
        # start = time.time()
        B, H, N, _ = dist.shape
        results = []
        maxC, maxS = 0, 0
        for b in range(B):
            row = []
            for h in range(H):
                mat = self.cluster_one_graph(dist[b, h], tau)
                row.append(mat)
                maxC = max(maxC, mat.shape[0])
                maxS = max(maxS, mat.shape[1])
            results.append(row)
        # Pad to tensor
        out = dist.new_full((B, H, maxC, maxS), -1, dtype=torch.long)
        for b in range(B):
            for h in range(H):
                mat = results[b][h]
                out[b, h, :mat.shape[0], :mat.shape[1]] = mat
        # torch.cuda.synchronize()
        # end = time.time()
        # print(f"[hard_leader_batched] Time on GPU: {end - start:.6f} seconds")
        
        return out
    def hard_leader_vectorized(self, dist, tau, keep_diag: bool = True):
        """
        Vectorized version of hard-leader clustering using distance threshold.
        This version is more GPU-friendly than the original, though not 100% vectorized.

        Args:
            dist: Tensor of shape [B, H, n, n]
            tau: Threshold for connection
            keep_diag: Whether to keep diagonal connections (not used)

        Returns:
            clusters_tensor: [B, H, max_num_clusters, max_cluster_size]
        """
        torch.cuda.synchronize()
        start = time.time()

        B, H, n, _ = dist.shape
        device = dist.device

        mask = (dist <= tau)

        all_clusters = []
        max_num_clusters = 0
        max_cluster_size = 0

        for b in range(B):
            batch_clusters = []
            for h in range(H):
                slice_mask = mask[b, h].clone()
                visited = torch.zeros(n, dtype=torch.bool, device=device)
                clusters = []

                # Cluster CLS token
                clusters.append([0])
                visited[0] = True

                # Build clusters greedily
                for i in range(1, n - 1):  # skip CLS/SEP
                    if visited[i]:
                        continue
                    group = torch.where(slice_mask[i] & ~visited)[0]
                    group = group[(group != 0) & (group != n - 1)]
                    group = torch.cat([group, torch.tensor([i], device=device)])  # FIXED LINE
                    group = torch.unique(group)
                    visited[group] = True
                    clusters.append(group.tolist())


                # Cluster SEP token
                clusters.append([n - 1])
                visited[n - 1] = True

                batch_clusters.append(clusters)
                max_num_clusters = max(max_num_clusters, len(clusters))
                max_cluster_size = max(
                    max_cluster_size,
                    max(len(c) for c in clusters) if clusters else 0
                )
            all_clusters.append(batch_clusters)

        # Preallocate output tensor
        clusters_tensor = torch.full(
            (B, H, max_num_clusters, max_cluster_size),
            fill_value=-1,
            dtype=torch.long,
            device=device
        )

        for b in range(B):
            for h in range(H):
                for i, group in enumerate(all_clusters[b][h]):
                    clusters_tensor[b, h, i, :len(group)] = torch.tensor(group, device=device)

        torch.cuda.synchronize()
        end = time.time()
        print(f"[Vectorized] Time on GPU: {end - start:.6f} seconds")
        return clusters_tensor
    def hard_leader_vectorized_deepSeek( self, dist, tau):
    
        """
        Optimized GPU version of hard-leader clustering.
        Identical logic to original CPU version, but 4-5x faster on GPU.
        
        Args:
            dist: Tensor of shape [B, H, n, n] containing pairwise distances
            tau: Threshold distance for cluster formation
            
        Returns:
            clusters_tensor: Tensor of shape [B, H, max_clusters, max_cluster_size]
                            with -1 padding for unused slots
        """
        torch.cuda.synchronize()
        start = time.time()
        B, H, n, _ = dist.shape
        device = dist.device
        
        # Vectorized mask creation
        mask = (dist <= tau)
        
        # Preallocate output tensor (conservative upper bounds)
        max_possible_clusters = 2 * n  # Worst case: each token forms its own cluster
        clusters_tensor = torch.full(
            (B, H, max_possible_clusters, n),
            fill_value=-1,
            dtype=torch.long,
            device=device
        )
        cluster_counts = torch.zeros(B, H, dtype=torch.long, device=device)

        # Initialize CLS (0) and SEP (n-1) clusters (vectorized)
        clusters_tensor[:, :, 0, 0] = 0
        clusters_tensor[:, :, 1, 0] = n - 1
        cluster_counts[:, :] = 2  # We've added 2 clusters (CLS and SEP)
        
        # Create visited mask
        visited = torch.zeros(B, H, n, dtype=torch.bool, device=device)
        visited[:, :, 0] = visited[:, :, -1] = True

        # Main processing loop (sequential in tokens but parallel in batches/heads)
        for i in range(1, n - 1):
            # Find unprocessed tokens across all batches/heads
            active = ~visited[:, :, i]
            active_indices = torch.where(active)
            
            if len(active_indices[0]) == 0:
                continue

            # Process each active token
            for b, h in zip(*active_indices):
                # Find connected unvisited nodes (same logic as original)
                neighbors = torch.where(mask[b, h, i] & ~visited[b, h])[0]
                neighbors = neighbors[(neighbors != 0) & (neighbors != n - 1)]
                
                # Get next available cluster slot
                c_idx = cluster_counts[b, h].item()
                
                if len(neighbors) > 0:
                    # Create new cluster
                    cluster = torch.cat([neighbors, torch.tensor([i], device=device)])
                    clusters_tensor[b, h, c_idx, :len(cluster)] = cluster
                    cluster_counts[b, h] += 1
                    visited[b, h, cluster] = True
                else:
                    # Single-token cluster
                    clusters_tensor[b, h, c_idx, 0] = i
                    cluster_counts[b, h] += 1
                    visited[b, h, i] = True

        # Trim unused clusters
        max_clusters = cluster_counts.max()
        torch.cuda.synchronize()
        end = time.time()
        print(f"[--DEEP SEEK--] Time on GPU: {end - start:.6f} seconds")
        return clusters_tensor[:, :, :max_clusters, :]
   
    def _process_clusters_kernel(self, mask, clusters_tensor, cluster_counts, n, device):
        B, H, n, _ = mask.shape
        
        # Initialize CLS/SEP clusters
        clusters_tensor[:, :, 0, 0] = 0
        clusters_tensor[:, :, 1, 0] = n - 1
        cluster_counts[:, :] = 2
        visited = torch.zeros(B, H, n, dtype=torch.bool, device=device)
        visited[:, :, 0] = visited[:, :, -1] = True
        for i in range(1, n - 1):
            active = ~visited[:, :, i]
            active_indices = torch.where(active)
            if len(active_indices[0]) == 0:
                continue
            for b, h in zip(*active_indices):
                neighbors = torch.where(mask[b, h, i] & ~visited[b, h])[0]
                neighbors = neighbors[(neighbors != 0) & (neighbors != n - 1)]
                c_idx = cluster_counts[b, h].item()
                if len(neighbors) > 0:
                    cluster = torch.cat([neighbors, torch.tensor([i], device=device)])
                    clusters_tensor[b, h, c_idx, :len(cluster)] = cluster
                    cluster_counts[b, h] += 1
                    visited[b, h, cluster] = True
                else:
                    clusters_tensor[b, h, c_idx, 0] = i
                    cluster_counts[b, h] += 1
                    visited[b, h, i] = True
                

    def hard_leader_gpu_DeepSeek2(self, dist, tau):
        # torch.cuda.synchronize()
        # start = time.time()
        # DeepSeek2_total = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage1 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage2 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage3 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage4 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage5 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_stage6 = torch.cuda.Event(enable_timing=True)
        # DeepSeek2_total.record()
        
        B, H, n, _ = dist.shape
        device = dist.device
        # DeepSeek2_stage1.record()
        mask = (dist <= tau)
        max_possible_clusters = 2 * n
        clusters_tensor = torch.full(
            (B, H, max_possible_clusters, n),
            fill_value=-1,
            dtype=torch.long,
            device=device
        )
        # DeepSeek2_stage2.record()
        cluster_counts = torch.zeros(B, H, dtype=torch.long, device=device)
        # DeepSeek2_stage3.record()
        self._process_clusters_kernel(mask, clusters_tensor, cluster_counts, n, device)
        # DeepSeek2_stage4.record()
        
        max_clusters = int(cluster_counts.max().item())
        # DeepSeek2_stage5.record()
        output = clusters_tensor[:, :, :max_clusters, :]
        # DeepSeek2_stage6.record()
        # torch.cuda.synchronize()
        # end = time.time()
        # print(f"[--DEEP SEEK2--] Time on GPU: {end - start:.6f} seconds")
        # torch.cuda.synchronize()

        # Print timings
        # print(f"DeepSeek2_stage 1 time: {DeepSeek2_total.elapsed_time(DeepSeek2_stage1):.3f} ms")
        # print(f"DeepSeek2_stage 2 time: {DeepSeek2_stage1.elapsed_time(DeepSeek2_stage2):.3f} ms")
        # print(f"DeepSeek2_stage 3 time: {DeepSeek2_stage2.elapsed_time(DeepSeek2_stage3):.3f} ms")
        # print(f"DeepSeek2_stage 4 time: {DeepSeek2_stage3.elapsed_time(DeepSeek2_stage4):.3f} ms")
        # print(f"DeepSeek2_stage 5 time: {DeepSeek2_stage4.elapsed_time(DeepSeek2_stage5):.3f} ms")
        # print(f"DeepSeek2_stage 6 time: {DeepSeek2_stage5.elapsed_time(DeepSeek2_stage6):.3f} ms")
        
        # print(f"Total time:   {DeepSeek2_total.elapsed_time(DeepSeek2_stage6):.3f} ms")
        return output
   
    def process_clusters_kernel_deterministic(self,mask, clusters_tensor, cluster_counts):
        B, H, n, _ = mask.shape
        clusters_tensor[:, :, 0, 0] = 0
        clusters_tensor[:, :, 1, 0] = n - 1
        cluster_counts.fill_(2)
        visited = torch.zeros((B, H, n), dtype=torch.bool, device=mask.device)
        visited[:, :, 0] = visited[:, :, -1] = True
        for i in range(1, n - 1):
            active_b, active_h = torch.where(~visited[:, :, i])
            order = torch.argsort(active_b * H + active_h)
            active_b, active_h = active_b[order], active_h[order]
            for b, h in zip(active_b.tolist(), active_h.tolist()):
                neigh = torch.where(mask[b, h, i] & ~visited[b, h])[0]
                neigh = neigh[(neigh != 0) & (neigh != n - 1)]
                c_idx = cluster_counts[b, h].item()
                if neigh.numel():
                    cluster = torch.cat((neigh, torch.tensor([i], device=mask.device)))
                    clusters_tensor[b, h, c_idx, :cluster.numel()] = cluster
                    visited[b, h, cluster] = True
                else:
                    clusters_tensor[b, h, c_idx, 0] = i
                    visited[b, h, i] = True
                cluster_counts[b, h] += 1

    def hard_leader_gpu_deepseek2_deterministic(self,dist, tau):
        B, H, n, _ = dist.shape
        device = dist.device
        mask = dist <= tau
        clusters_tensor = torch.full((B, H, 2 * n, n), -1, dtype=torch.long, device=device)
        cluster_counts = torch.zeros((B, H), dtype=torch.long, device=device)
        self.process_clusters_kernel_deterministic(mask, clusters_tensor, cluster_counts)
        return clusters_tensor, cluster_counts
# ------------------------------------------------------------------

   
   
    def apply_clustered_row_averaging(self, x, clusters):
 
        """
        Applies row-averaging over clusters using fast, vectorized GPU ops.
        
        Args:
            x: Tensor of shape [B, C, n, n] on GPU
            clusters: list of lists of lists [B][C][num_clusters][indices]

        Returns:
            Averaged tensor of same shape as x
        """
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        B, C, n, _ = x.shape
        device = x.device
        out = x.clone()

        for b in range(B):
            for c in range(C):
                mat = x[b, c]  # [n, n]
                out_mat = out[b, c]
                
                # Preallocate buffers
                summed = torch.zeros_like(mat)
                counts = torch.zeros(n, device=device)

                for group in clusters[b][c]:
                    idx = torch.tensor(group, device=device, dtype=torch.long)
                    if idx.numel() == 0:
                        continue
                    mean_row = mat.index_select(0, idx).mean(dim=0)
                    summed.index_add_(0, idx, mean_row.expand(idx.size(0), -1))
                    counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))

                # Avoid division by zero
                nonzero = counts > 0
                out_mat[nonzero] = summed[nonzero] / counts[nonzero].unsqueeze(1)
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'apply_clustered_row_averaging' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return out
    def apply_clustered_row_averaging_vectorized(self,x, clusters):
        
        """
        Vectorized version: averages over token rows in each cluster.

        Args:
            x: Tensor of shape [B, H, T, D]
            clusters: LongTensor of shape [B, H, num_clusters, cluster_size], padded with -1

        Returns:
            out: Tensor of same shape as x, where clustered rows have been averaged
        """
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        assert x.ndim == 4, f"x must be [B, H, T, D], got {x.shape}"
        assert clusters.ndim == 4, f"clusters must be [B, H, num_clusters, k], got {clusters.shape}"

        B, H, T, D = x.shape
        _, _, num_clusters, k = clusters.shape
        device = x.device

        x_flat = x.reshape(B * H, T, D)
        clusters_flat = clusters.reshape(B * H, num_clusters, k)
        mask = clusters_flat != -1                                # valid entries
        clusters_safe = clusters_flat.clamp(min=0)

        # Build flat index selectors
        bc_indices = torch.arange(B * H, device=device).view(-1, 1, 1).expand(-1, num_clusters, k)
        selected_rows = x_flat[bc_indices, clusters_safe]         # [BH, num_clusters, k, D]
        selected_rows = selected_rows * mask.unsqueeze(-1)        # zero out invalid rows

        # Compute means
        summed = selected_rows.sum(dim=2)                         # [BH, num_clusters, D]
        counts = mask.sum(dim=2).clamp(min=1).unsqueeze(-1)       # [BH, num_clusters, 1]
        mean_rows = summed / counts                               # [BH, num_clusters, D]

        # Repeat mean rows back to original cluster locations
        repeated_means = mean_rows.unsqueeze(2).expand(-1, -1, k, -1)  # [BH, num_clusters, k, D]
        flat_rows = repeated_means[mask]                               # [N, D]
        flat_targets = clusters_safe[mask]                             # [N]
        flat_bc = bc_indices[mask]                                     # [N]

        # Write averaged rows
        out_flat = x_flat.clone()
        out_flat.index_put_((flat_bc, flat_targets), flat_rows, accumulate=False)

        # Reshape back to [B, H, T, D]
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'apply_clustered_row_averaging_vectorized' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return out_flat.view(B, H, T, D)


    
    def clustering_stats(self, clusters_batch):
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        total_clustered_count = 0
        total_original_count = 0
        # print("clusters_batch",clusters_batch)
        for batch in clusters_batch:
            for channel_clusters in batch:
                all_indices = [i for group in channel_clusters for i in group]
                n = len(set(all_indices))  # counts unique indices, i.e. number of surviving/pruned rows
                n_clusters = len(channel_clusters)
                total_original_count += n
                total_clustered_count += n_clusters
                # print("set(all_indices)",set(all_indices))
                # print("n",n)
                # print("n_clusters",n_clusters)
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'clustering_stats' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return total_clustered_count, total_original_count
   


    def avg_unique_nonpad(self, x: torch.Tensor, pad_val: int = -1):
        """
        Average number of unique non-pad values per row/tensor,
        skipping rows that are fully pad OR that have only a single unique non-pad value.
        """
        # Flatten all but the last dimension (treat each last-dim row as one 'tensor')
        rows = x.reshape(-1, x.size(-1))

        counts = []
        for row in rows:
            valid = row[row != pad_val]          # drop pads
            if valid.numel() == 0:
                continue                         # skip fully padded rows
            uniq = torch.unique(valid)
            if uniq.numel() <= 1:
                continue                         # skip rows with only one unique non-pad value
            counts.append(int(uniq.numel()))
        global Totalsum
        global TotalCounts
        
        Totalsum+= sum(counts)
        TotalCounts+= len(counts)
        
        print("Totalsum",Totalsum)
        print("TotalCounts",TotalCounts)
        
        
        return float(sum(counts) / len(counts)) if counts else 0.0
       
        
        
        
        
    
    def dedup_per_row(self,x: torch.Tensor, pad_val: int = -1, dim: int = -1) -> torch.Tensor:
        """
        Remove duplicates within each row along `dim` without loops.
        Keeps the same shape by left-compacting uniques and padding with `pad_val`.

        x: LongTensor with padding == pad_val (e.g., -1). Shape [..., L] along `dim`.
        """
        # Move the target dim to last
        x_perm = x.transpose(dim, -1).contiguous()          # [..., L]
        *batch, L = x_perm.shape
        flat = x_perm.view(-1, L)                           # [N, L]

        # Mark valid entries and push pads to a large value so they sort to the end
        valid = flat != pad_val
        maxv = torch.iinfo(flat.dtype).max
        tmp = flat.masked_fill(~valid, maxv)

        # Row-wise sort => duplicates become consecutive; pads at the end
        sorted_vals, _ = torch.sort(tmp, dim=1)

        # Identify pads and duplicates (equal to previous element)
        is_pad = (sorted_vals == maxv)
        dup = torch.zeros_like(sorted_vals, dtype=torch.bool)
        dup[:, 1:] = (sorted_vals[:, 1:] == sorted_vals[:, :-1])

        # Keep only first occurrence of each value (and never keep pads)
        keep = (~dup) & (~is_pad)

        # Compute target positions for the kept elements (0..k-1 per row)
        pos = keep.cumsum(dim=1) - 1                        # -1 where keep==False

        # Scatter kept values into compacted output, pad the rest
        out = torch.full_like(sorted_vals, pad_val)
        rows = torch.arange(out.size(0), device=out.device).unsqueeze(1).expand_as(out)
        out[rows[keep], pos[keep]] = sorted_vals[keep]

        # Restore original shape and dim order
        out = out.view(*batch, L)
        out = out.transpose(dim, -1).contiguous()
        return out

    def clustering_stats_vectorized(self,clusters):
        """
        Computes clustering stats from padded tensor.

        Args:
            clusters: LongTensor of shape [B, C, num_clusters, cluster_size], with -1 padding

        Returns:
            total_clustered_count: Total number of non-empty clusters
            total_original_count: Total number of unique token indices across all clusters
        """
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        # print("clusters",clusters)
        #remove duplications in rows
        clusters=self.dedup_per_row(clusters)
        B, C, num_clusters, cluster_size = clusters.shape
        device = clusters.device
        # print("B, C, num_clusters",B, C, num_clusters)
        # Step 1: Flatten cluster structure
        flat_clusters = clusters.view(-1, cluster_size)  # [B*C*num_clusters, cluster_size]
        # print("flat_clusters",flat_clusters)
        # Step 2: Mask valid indices
        valid_mask = flat_clusters != -1
        valid_indices = flat_clusters[valid_mask]  # [N]
        # print("valid_indices",valid_indices)
        # Step 3: Count non-pad entries (i.e., just how many values remain)
        total_original_count = int(valid_indices.numel())               # simplest   

        # print("total_original_count",total_original_count)

        # Step 4: Count non-empty clusters (rows with any valid index)
        non_empty_clusters = valid_mask.any(dim=1)  # [B*C*num_clusters]
        total_clustered_count = non_empty_clusters.sum().item()
        # print("non_empty_clusters",non_empty_clusters)
        # print("total_clustered_count",total_clustered_count)
        # exit
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'clustering_stats_vectorized' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return total_clustered_count, total_original_count
    
    def filter_clusters_by_zero_indices(self,clusters, mask):
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        mask  = mask.any(dim=-1)  # shape: [1, 12, 12]
        # print("mask inside filter_clusters_by_zero_indices", mask)
        B, H, n = mask.shape
        filtered_clusters = []
        for b in range(B):
            batch_clusters = []
            for h in range(H):
                head_clusters = []
                for group in clusters[b][h]:
                    # Only keep indices where mask == True
                    kept_group = [idx for idx in group if mask[b, h, idx]]
                    if kept_group:  # skip empty clusters
                        head_clusters.append(kept_group)
                batch_clusters.append(head_clusters)
            filtered_clusters.append(batch_clusters)
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'filter_clusters_by_zero_indices' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return filtered_clusters
       

    
    def filter_clusters_by_zero_indices_vectorized(self,clusters, mask):

        """
        Fully vectorized version (no Python loops).
        Filters out invalid indices from clusters using a mask.

        Args:
            clusters: LongTensor of shape [B, H, num_clusters, cluster_size] with -1 padding
            mask: BoolTensor of shape [B, H, n] or [B, H, n, k]

        Returns:
            new_filtered: LongTensor of shape [B, H, max_kept_clusters, cluster_size] with -1 padding
        """
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        B, H, num_clusters, cluster_size = clusters.shape
        device = clusters.device

        # Step 0: Handle 4D mask case
        if mask.ndim == 4:
            mask = mask.any(dim=-1)  # reduce to [B, H, n]

        # Step 1: Clamp -1 indices for safe indexing
        valid_mask = clusters >= 0
        clamped_clusters = clusters.clamp(min=0)  # [B, H, num_clusters, cluster_size]

        # Step 2: Gather validity of each index from mask
        # mask: [B, H, n] → [B, H, num_clusters, n]
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, num_clusters, -1)
        index_mask = torch.gather(mask_expanded, 3, clamped_clusters)  # [B, H, num_clusters, cluster_size]

        # Step 3: Combine with padding mask
        combined_mask = valid_mask & index_mask
        filtered = clusters.masked_fill(~combined_mask, -1)  # Set all invalid entries to -1

        # Step 4: Identify non-empty clusters
        cluster_nonempty = (filtered != -1).any(dim=-1)  # [B, H, num_clusters]
        max_clusters = cluster_nonempty.sum(dim=-1).max().item()  # scalar

        # Step 5: Flatten [B, H] → [B*H]
        filtered_flat = filtered.view(B * H, num_clusters, cluster_size)
        cluster_nonempty_flat = cluster_nonempty.view(B * H, num_clusters)

        # Step 6: Get indices of non-empty clusters
        keep_idx = cluster_nonempty_flat.nonzero(as_tuple=False)  # [K, 2]
        batch_head_idx = keep_idx[:, 0]
        cluster_idx = keep_idx[:, 1]

        # Step 7: Allocate new padded tensor
        new_filtered = torch.full(
            (B * H, max_clusters, cluster_size),
            fill_value=-1,
            dtype=torch.long,
            device=device
        )

        # Step 8: Scatter clusters to new padded output (still looped, but vector-safe)
        insert_pos = torch.zeros(B * H, dtype=torch.long, device=device)
        for i in range(len(keep_idx)):
            bh = batch_head_idx[i]
            idx = insert_pos[bh].item()
            new_filtered[bh, idx] = filtered_flat[bh, cluster_idx[i]]
            insert_pos[bh] += 1

        # Step 9: Reshape back to [B, H, max_kept_clusters, cluster_size]
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'filter_clusters_by_zero_indices_vectorized' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        return new_filtered.view(B, H, max_clusters, cluster_size)
    
   

    def filter_clusters_by_zero_indices_vectorized_noLOOP(self,clusters, mask):
        """
        GPU-optimized, loop-free version of cluster filtering using torch.scatter.
        
        Args:
            clusters: LongTensor [B, H, num_clusters, cluster_size] with -1 padding
            mask:     BoolTensor [B, H, n] or [B, H, n, k]
            
        Returns:
            new_filtered: LongTensor [B, H, max_kept_clusters, cluster_size] with -1 padding
        """
        # torch.cuda.synchronize()  # Wait for prior GPU ops
        # start = time.time()
        
        B, H, num_clusters, cluster_size = clusters.shape
        device = clusters.device

        # Step 0: Collapse 4D mask if needed
        if mask.ndim == 4:
            mask = mask.any(dim=-1)  # [B, H, n]

        # Step 1: Clamp invalid indices and cast to long
        valid_mask = clusters >= 0
        clamped_clusters = clusters.clamp(min=0).long()

        # Step 2: Gather mask values for indices in each cluster
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, num_clusters, -1)
        index_mask = torch.gather(mask_expanded, 3, clamped_clusters)

        # Step 3: Apply filtering
        combined_mask = valid_mask & index_mask
        filtered = clusters.masked_fill(~combined_mask, -1).long()

        # Step 4: Identify non-empty clusters
        cluster_nonempty = (filtered != -1).any(dim=-1)  # [B, H, num_clusters]
        num_kept = cluster_nonempty.sum(dim=-1)          # [B, H]
        max_clusters = num_kept.max().item()

        # Step 5: Flatten for scatter operation
        BH = B * H
        filtered_flat = filtered.view(BH, num_clusters, cluster_size)
        nonempty_flat = cluster_nonempty.view(BH, num_clusters)

        # Step 6: Gather non-empty indices
        keep_idx = nonempty_flat.nonzero(as_tuple=False)  # [K, 2]
        bh_indices = keep_idx[:, 0]
        cluster_indices = keep_idx[:, 1]

        # Step 7: Determine insert positions
        insert_counts = torch.bincount(bh_indices, minlength=BH)
        insert_positions = torch.cat([
            torch.arange(n, device=device) if n > 0 else torch.empty(0, dtype=torch.long, device=device)
            for n in insert_counts.tolist()
        ])

        # Step 8: Scatter filtered values into new padded tensor
        new_filtered = torch.full(
            (BH, max_clusters, cluster_size),
            -1,
            dtype=torch.long,
            device=device
        )
        new_filtered[bh_indices, insert_positions] = filtered_flat[bh_indices, cluster_indices]

        # Step 9: Reshape to original batch layout
        # torch.cuda.synchronize()
        # end = time.time()
        # print("time in 'filter_clusters_by_zero_indices_vectorized_noLOOP' =", f"{end - start:.6f} seconds")

        return new_filtered.view(B, H, max_clusters, cluster_size)

    def Prune_Query(self, Query_layer):
        
        # print("Query_layer shape",Query_layer.shape)
        # print("Query_layer ",Query_layer)
        #get the integer part of the Query_layer
        Query_layer_MSB = Query_layer/2**MSBFirstround
        Query_layer_MSB = torch.trunc(Query_layer_MSB)
        # print("Query_layer_MSB ",Query_layer_MSB)
        Query_layer_Fractions = Query_layer - Query_layer_MSB
        
        #get MSBits
        # msbN = torch.floor(Query_layer_MSB / 2**HyperParameters.MSBits )
        # print("msbN ",msbN)
        
        msbN = Query_layer
        
        
        
        s_mean = msbN.mean(dim=-1, keepdim=True)   # [B,H,N,1]
        s_max  = msbN.amax(dim=-1, keepdim=True)   # [B,H,N,1]
        s_min  = msbN.amin(dim=-1, keepdim=True)   # [B,H,N,1]
        # print("s_mean ",s_mean)
        # print("s_max ",s_max)
        # print("s_min ",s_min)
        
        #To find the Query pruning threshold
        # print("HyperParameters.QueryPrRatio",HyperParameters.QueryPrRatio)
        if( HyperParameters.QueryPrRatio>=0 and HyperParameters.QueryPrRatio<=1):
            threshold=torch.add(torch.mul(s_max,HyperParameters.QueryPrRatio) , (torch.mul(s_mean,(1-HyperParameters.QueryPrRatio)) ))
        elif ( HyperParameters.QueryPrRatio>=-1 and HyperParameters.QueryPrRatio<0 ):
            threshold=torch.add(torch.mul(s_min,-HyperParameters.QueryPrRatio) ,(torch.mul(s_mean,(1+HyperParameters.QueryPrRatio))))
        # print("threshold ",threshold)    
               
            
        mask_keep = msbN >= threshold
        
        mask_keep[:, :, 0]  = True   # keep CLS
        mask_keep[:, :, -1] = True   # keep SEP
        # print("mask_keep ",mask_keep)
        pruned = Query_layer * mask_keep.to(Query_layer.dtype)
        # print("pruned ",pruned)
        return pruned 
        
    def Prune_Query_N_M(self, Query_layer, n_per_block: int = 1, block_size: int = 8):
        """
        Drop the n smallest values per block (contiguous chunks of `block_size`)
        along the last dimension. Keeps [CLS]=idx 0 and [SEP]=idx -1.
    
        Args:
            Query_layer: tensor [..., N] (e.g., [B, H, N]); N must be multiple of block_size.
            n_per_block: how many smallest values to drop *per block* (e.g., 1, 2, 3).
            block_size:  block width (default 8).
        Returns:
            pruned: tensor of same shape as Query_layer with dropped entries zeroed.
        """
        
        block_size = HyperParameters.QueryBlock
        n_per_block = HyperParameters.n_per_Queryblock
        # print("Query_layer",Query_layer)
        
        
        # Query_layer_MSB = Query_layer/2**MSBFirstround
        # Query_layer_MSB = torch.trunc(Query_layer_MSB)
        # # print("Query_layer_MSB ",Query_layer_MSB)
        # Query_layer_Fractions = Query_layer - Query_layer_MSB
        
        #get MSBits
        # msbN = torch.floor(Query_layer_MSB / 2**HyperParameters.MSBits )
        # print("msbN ",msbN)
        
        #msbN = Query_layer_MSB


        x = Query_layer
        *prefix, N = x.shape
        assert N % block_size == 0, f"N={N} must be a multiple of block_size={block_size}"
        num_blocks = N // block_size
    
        # === Scores to rank by (use x for "least value"; use x.abs() for "least magnitude") ===
        scores = x.abs()
    
        # Exclude [CLS] and [SEP] from being dropped by giving them +inf during selection
        INF = torch.finfo(scores.dtype).max if scores.is_floating_point() else torch.iinfo(scores.dtype).max
        sel_scores = scores.clone()
        sel_scores[:, :, 0] = INF   # keep CLS
        sel_scores[:, :, -1] = INF   # keep SEP
    
        # Reshape to blocks: [..., num_blocks, block_size]
        s = sel_scores.reshape(*prefix, num_blocks, block_size)
    
        k = max(0, min(int(n_per_block), block_size))  # topk expects a fixed k
        if k > 0:
            # Get the k SMALLEST per block (values + indices). largest=False -> smallest.  :contentReference[oaicite:0]{index=0}
            vals, idx = torch.topk(s, k=k, dim=-1, largest=False, sorted=False)
    
            # Build a drop mask per block; ignore any selection that hit +inf (i.e., CLS/SEP)
            is_finite = torch.isfinite(vals) if vals.is_floating_point() else (vals != INF)  #  :contentReference[oaicite:1]{index=1}
            drop_blocks = torch.zeros_like(s, dtype=torch.bool)
            drop_blocks.scatter_(-1, idx, is_finite)  # place True at chosen (finite) indices  :contentReference[oaicite:2]{index=2}
        else:
            drop_blocks = torch.zeros_like(s, dtype=torch.bool)
    
        # Back to original shape
        drop_mask = drop_blocks.reshape(*prefix, N)
    
        # Enforce hard keep for CLS/SEP
        drop_mask[:, :, 0]  = False
        drop_mask[:, :, -1] = False
    
        keep_mask = ~drop_mask
        # print("keep_mask",keep_mask)
        pruned = x * keep_mask.to(x.dtype)
        # print("pruned",pruned)
        return pruned    
        
    def Prune_Keys(self, key_layer):
        # torch.cuda.synchronize()  # Wait for all prior GPU tasks to finish
        # start = time.time()
        # Prune_Keys_total = torch.cuda.Event(enable_timing=True)
        # Prune_Keys_stage1 = torch.cuda.Event(enable_timing=True)
        # Prune_Keys_stage2 = torch.cuda.Event(enable_timing=True)
        # Prune_Keys_stage3 = torch.cuda.Event(enable_timing=True)
        # Prune_Keys_stage4 = torch.cuda.Event(enable_timing=True)
                
        # Prune_Keys_total.record()
        
        key_dim=key_layer.shape
        # print("key_dim",key_dim)
        # print("key_layer")
        # print(key_layer)     
        # global MSBFirstround
        
        #get the integer part of the Key
        key_layer_MSB=key_layer/2**MSBFirstround
        key_layer_MSB=torch.trunc(key_layer_MSB)
       
        # print("key_layer_MSB,shape")
        # print(key_layer_MSB.shape)
        # print("key_layer_MSB")
        # print(key_layer_MSB)
        #get the fraction part of the Key
        key_layer_Fractions=key_layer-key_layer_MSB
        #get MSBits
        msbN = torch.floor(key_layer_MSB / 2**HyperParameters.MSBits )          
        # print("key_layer_msbN")
        # print(msbN)
        
        
        #-------------------------------------------------------
        #-------------------------------------------------------
        #call Importance function
        #-------------------------------------------------------
        #-------------------------------------------------------
        Importance = self.compute_importance_l1(msbN)
        # Prune_Keys_stage1.record()
        # print("Importance.shape")
        # print(Importance.shape)
        # print("Importance.")
        # print(Importance)
        
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        #Call the cluster algorithm
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        clusters = self.hard_leader_gpu_DeepSeek2(Importance, HyperParameters.tau)
        # Prune_Keys_stage2.record()
       
        
        #Importance = self.compute_importance_l2(key_layer_MSB)
        
        S_mean=torch.mean(Importance,(3))
        # print("S_mean.shape")
        # print(S_mean.shape)
        # print("S_mean.")
        # print(S_mean)
        S_max=torch.max(S_mean, dim=2, keepdim=True)[0]
        # print("S_max.shape")
        # print(S_max.shape)
        # print("S_max.")
        # print(S_max)
        S_min=torch.min(S_mean, dim=2, keepdim=True)[0]
        # print("S_min.shape")
        # print(S_min.shape)
        # print("S_min.")
        # print(S_min)
        S_mean2=torch.mean(S_mean, dim=2, keepdim=True)[0]
        # print("S_mean2.shape")
        # print(S_mean2.shape)
        # print("S_mean2.")
        # print(S_mean2)
        # Prune_Keys_stage3.record()
        #-----------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------------------------
        #To find the pruning threshold
        if( HyperParameters.PruningRatio>=0 and HyperParameters.PruningRatio<=1):
            threshold=torch.add(torch.mul(S_max,HyperParameters.PruningRatio) , (torch.mul(S_mean2,(1-HyperParameters.PruningRatio)) ))
        elif ( HyperParameters.PruningRatio>=-1 and HyperParameters.PruningRatio<0 ):
            threshold=torch.add(torch.mul(S_min,-HyperParameters.PruningRatio) ,(torch.mul(S_mean2,(1+HyperParameters.PruningRatio))))
        #-----------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------------------------
        
        #To find the KEEP_Threshold
        if( HyperParameters.KeepRatio>=0 and HyperParameters.KeepRatio<=1):
            threshold_KEEP=torch.add(torch.mul(S_max,HyperParameters.KeepRatio) , (torch.mul(S_mean2,(1-HyperParameters.KeepRatio)) ))
        elif ( HyperParameters.KeepRatio>=-1 and HyperParameters.KeepRatio<0 ):
            threshold_KEEP=torch.add(torch.mul(S_min,-HyperParameters.KeepRatio) ,(torch.mul(S_mean2,(1+HyperParameters.KeepRatio))))
        #-----------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------------------------
                    
        # print("threshold_KEEP.shape")
        # print(threshold_KEEP.shape)
        # print("threshold_KEEP.")
        # print(threshold_KEEP)
        threshold_expanded = threshold.expand_as(S_mean)
        KEEP_threshold_expanded = threshold_KEEP.expand_as(S_mean)
        # print("KEEP_threshold_expanded.shape")
        # print(KEEP_threshold_expanded.shape)
        # print("KEEP_threshold_expanded.")
        # print(KEEP_threshold_expanded)
        # Step 1: Create a modified S_mean where [CLS] and [SEP] are set to +inf (always kept)
        S_mean_modified = S_mean.clone()  # Avoid modifying original tensor
        S_mean_modified[:, :, 0] = float('inf')    # [CLS] will always pass threshold
        S_mean_modified[:, :, -1] = float('inf')   # [SEP] will always pass threshold
        # print("S_mean_modified")
        # print(S_mean_modified)
        
        #############################################
        #to prune least important%
        mask=torch.le(S_mean_modified, threshold_expanded)
        mask = ~mask
        ############################################
        #to prune Most important %
        #mask=torch.gt(S_mean_modified, threshold_expanded)
        #mask = ~mask
        ########################################################
        #KEEP any VAl larger than KEEP Threshold
        KEEP_MASK = torch.gt(S_mean_modified, KEEP_threshold_expanded)
        # print("KEEP_MASK.shape")
        # print(KEEP_MASK.shape)
        # print("KEEP_MASK.")
        # print(KEEP_MASK)
        mask[:, :, 0] = True              # Keep CLS
        mask[:, :, -1] = True             # Keep SEP
        # Repeat the last dimension 
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, key_dim[3])  
        KEEP_MASK = KEEP_MASK.unsqueeze(-1).repeat(1, 1, 1, key_dim[3])
        
        # print("KEEP_MASK.shape")
        # print(KEEP_MASK.shape)
        # print("KEEP_MASK.")
        # print(KEEP_MASK)
        # exit
        zero_indices = mask == False
        # print("Prune mask.shape")
        # print(mask.shape)
        # print("Prune mask.")
        # print(mask)
        result = key_layer* mask
        # print("result.shape")
        # print(result.shape)
        # print("result.")
        # print(result)
        # exit
        # torch.cuda.synchronize()  # Wait for matmul to finish
        # end = time.time() 
        # print("time in 'Prune_Keys' Fun =")
        # print(f"Time on GPU: {end - start:.6f} seconds")
        # Prune_Keys_stage4.record()
        # torch.cuda.synchronize()

        # Print timings
        # print(f"Prune_Keys_stage 1 time: {Prune_Keys_total.elapsed_time(Prune_Keys_stage1):.3f} ms")
        # print(f"Prune_Keys_stage 2 time: {Prune_Keys_stage1.elapsed_time(Prune_Keys_stage2):.3f} ms")
        # print(f"Prune_Keys_stage 3 time: {Prune_Keys_stage2.elapsed_time(Prune_Keys_stage3):.3f} ms")
        # print(f"Prune_Keys_stage 4 time: {Prune_Keys_stage3.elapsed_time(Prune_Keys_stage4):.3f} ms")
        
        # print(f"Total time:   {Prune_Keys_total.elapsed_time(Prune_Keys_stage4):.3f} ms")
        return result,mask,KEEP_MASK,clusters
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        query_layer=torch.round(query_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        query_layer=torch.clip(query_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        key_layer=torch.round(key_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        key_layer=torch.clip(key_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        ##CAll the prune KEY Function
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        # print("KEY Shape BEFORE",key_layer.shape)
        # print("KEY BEFORE",key_layer)
        # start_total = torch.cuda.Event(enable_timing=True)
        # after_stage1 = torch.cuda.Event(enable_timing=True)
        # after_stage2 = torch.cuda.Event(enable_timing=True)
        # after_stage3 = torch.cuda.Event(enable_timing=True)
        # after_stage4 = torch.cuda.Event(enable_timing=True)
        # after_stage5 = torch.cuda.Event(enable_timing=True)
        # after_stage6 = torch.cuda.Event(enable_timing=True)
        
        # start_total.record()
        PrunedKEYS,mask,KEEP_MASK,clusters = self.Prune_Keys(key_layer)
        # after_stage1.record()
        # print("Pruning MASK shape",mask.shape)
        # print("Pruning MASK",mask)
        # print("KEEP_MASK shape",KEEP_MASK.shape)
        # print("KEEP_MASK",KEEP_MASK)
        # print("KEY Pruned Shape After",PrunedKEYS.shape)
        # print("KEY Pruned After",PrunedKEYS)
        
        # print("Clusters",clusters)
        #remove pruned indices from the clustering LIST
        filtered_clusters = self.filter_clusters_by_zero_indices_vectorized_noLOOP(clusters, mask)
        # after_stage2.record()
        # print("filtered_clusters After Removing Pruned rows",filtered_clusters)
        #remove KEEP indicies from clustering List
        filtered_clusters = self.filter_clusters_by_zero_indices_vectorized_noLOOP(filtered_clusters, ~KEEP_MASK)
        #avg = self.avg_unique_nonpad(filtered_clusters)
        
        
        # after_stage3.record()
        # print("filtered_clusters After Removing KEEP rows",filtered_clusters)
        #filtered_clusters = filtered_clusters.float()
        KEY_clusteres = self.apply_clustered_row_averaging_vectorized(PrunedKEYS,filtered_clusters)
        # after_stage4.record()
        # print("KEY_clusteres Shape AFTER",KEY_clusteres.shape)
        # print("KEY_clusteres AFTER ",KEY_clusteres)
        
        key_layer = KEY_clusteres
        #find clustering Percent
        Cluster_Count_current,original_Count_current = self.clustering_stats_vectorized(filtered_clusters)
        # after_stage5.record()
        # exit
        HyperParameters.Cluster_Count = HyperParameters.Cluster_Count + Cluster_Count_current
        HyperParameters.original_Count = HyperParameters.original_Count + original_Count_current
        # print("HyperParameters.Cluster_Count",HyperParameters.Cluster_Count)
        
        # print("value_layer Shape BEFORE",value_layer.shape)
        # print("value_layer BEFORE",value_layer)
        #prune Value with the same mask as KEY
        value_layer = value_layer * mask 
        # print("value_layer Pruned Shape After",value_layer.shape)
        # print("value_layer Pruned After",value_layer)
        #cluster the Values
        value_layer = self.apply_clustered_row_averaging_vectorized(value_layer,filtered_clusters)
        # after_stage6.record()
        # print("value_layer_clusteres Shape AFTER",value_layer.shape)
        # print("value_layer_clusteres AFTER ",value_layer)
        # exit
        # Wait for all events to complete
        # torch.cuda.synchronize()

        # # Print timings
        # print(f"Stage 1 time: {start_total.elapsed_time(after_stage1):.3f} ms")
        # print(f"Stage 2 time: {after_stage1.elapsed_time(after_stage2):.3f} ms")
        # print(f"Stage 3 time: {after_stage2.elapsed_time(after_stage3):.3f} ms")
        # print(f"Stage 4 time: {after_stage3.elapsed_time(after_stage4):.3f} ms")
        # print(f"Stage 5 time: {after_stage4.elapsed_time(after_stage5):.3f} ms")
        # print(f"Stage 6 time: {after_stage5.elapsed_time(after_stage6):.3f} ms")
        # print(f"Total time:   {start_total.elapsed_time(after_stage6):.3f} ms")
        value_layer=torch.round(value_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        value_layer=torch.clip(value_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        #############################
        #apply to Q 
        ###############################
        #-----------------
        if(HyperParameters.ApplyTO_Q == 1):
            query_layer = query_layer * mask
            query_layer = self.apply_clustered_row_averaging_vectorized(query_layer,filtered_clusters)
            query_layer=torch.round(query_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
            query_layer=torch.clip(query_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        else:
            query_layer=query_layer
        #-----------------
        ###############################################
        ###############################################
        #Prune Query based on the query pruning threshold
        # print("query_layer Before",query_layer)
        query_layer = self.Prune_Query_N_M(query_layer)
        # print("query_layer After",query_layer)
        # exit
        ###############################################
        ###############################################
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores=torch.round(attention_scores*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        attention_scores=torch.clip(attention_scores,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores=torch.round(attention_scores*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        attention_scores=torch.clip(attention_scores,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) 
        attention_probs=torch.round(attention_probs*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        attention_probs=torch.clip(attention_probs,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer=torch.round(context_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        context_layer=torch.clip(context_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        #print("eager")
        return outputs


class BertSdpaSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once implemented.
            logger.warning_once(
                "BertSdpaSelfAttention is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "non-absolute `position_embedding_type` or `output_attentions=True` or `head_mask`. Falling back to "
                "the manual attention implementation, but specifying the manual implementation will be required from "
                "Transformers version v5.0.0 onwards. This warning can be removed using the argument "
                '`attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )
        query_layer=torch.round(query_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        query_layer=torch.clip(query_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        key_layer=torch.round(key_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        key_layer=torch.clip(key_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        value_layer=torch.round(value_layer*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        value_layer=torch.clip(value_layer,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        
        attn_output=torch.round(attn_output*(2**HyperParameters.fractionsFXP))/(2**HyperParameters.fractionsFXP)
        attn_output=torch.clip(attn_output,min=HyperParameters.MinFXP,max=HyperParameters.MaxFXP)
        
        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        print("SDPA")
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


BERT_SELF_ATTENTION_CLASSES = {
    "eager": BertSelfAttention,
    "sdpa": BertSdpaSelfAttention,
}


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BERT_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


@auto_docstring
class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, BertLMPredictionHead):
            module.bias.data.zero_()


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    seq_relationship_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@auto_docstring(
    custom_intro="""
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
)
class BertModel(BertPreTrainedModel):
    _no_split_modules = ["BertEmbeddings", "BertLayer"]

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """
)
class BertForPreTraining(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
            pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert Model with a `language modeling` head on top for CLM fine-tuning.
    """
)
class BertLMHeadModel(BertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **loss_kwargs,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(prediction_scores, labels, self.config.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@auto_docstring
class BertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @classmethod
    def can_generate(cls) -> bool:
        """
        Legacy correction: BertForMaskedLM can't call `generate()` from `GenerationMixin`, even though it has a
        `prepare_inputs_for_generation` method.
        """
        return False


@auto_docstring(
    custom_intro="""
    Bert Model with a `next sentence prediction (classification)` head on top.
    """
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "BertForMaskedLM",
    "BertForMultipleChoice",
    "BertForNextSentencePrediction",
    "BertForPreTraining",
    "BertForQuestionAnswering",
    "BertForSequenceClassification",
    "BertForTokenClassification",
    "BertLayer",
    "BertLMHeadModel",
    "BertModel",
    "BertPreTrainedModel",
    "load_tf_weights_in_bert",
]
