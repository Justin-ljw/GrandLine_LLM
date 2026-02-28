"""
训练工具函数集合
"""
import os
import sys
__package__ = "train"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


# 多 GPU 分布式训练中，判断当前进程是否为主进程（rank 0），用于控制日志输出和模型保存等操作
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# 简单的日志函数，只有主进程会输出日志，避免多 GPU 训练时日志重复
def Logger(content):
    if is_main_process():
        print(content)
        

def get_lr(lr, current_step, total_steps, warmup_steps=0):
    """
    学习率调度器：Warmup + Cosine Decay
    - warmup_steps: 线性warmup的步数
    - 之后使用cosine decay衰减
    """
    if current_step < warmup_steps:
        # 线性warmup: 从 0 增长到 lr
        return lr * (current_step / warmup_steps)
    else:
        # Cosine decay: 从 lr 衰减到 0.1*lr
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
    

class SkipBatchSampler(Sampler):
    '''
    自定义 BatchSampler，支持跳过前 skip 个样本，适用于训练恢复时的情况
    '''
    def __init__(self, sampler, batch_size: int, skip_batches: int=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches
        
    def __iter__(self):
        batch = []
        skipped = 0
        
        for idx in self.sampler:
            batch.append(idx)
            # 当积累的样本数量达到 batch_size 时输出
            if len(batch) == self.batch_size:
                # 跳过前 skip_batches 个批次
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                
                yield batch
                batch = []
        
        # 最后一个批次的样本数可能不足 batch_size，但仍然需要输出（如果没有被跳过）
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
    
    # Sampler 的长度要减去被跳过的批次数量
    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
                
                