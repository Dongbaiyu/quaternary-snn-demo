from __future__ import annotations

import torch
import torch.nn as nn


SURROGATE_ALPHA = 0.5


class TrainableQuaternarySpikeAct(torch.autograd.Function):
    """Exact quaternary spike and surrogate gradient from qua_v10.py."""

    @staticmethod
    def forward(ctx, input_norm):
        ctx.save_for_backward(input_norm)
        spikes = torch.zeros_like(input_norm)
        spikes[input_norm > 1.5] = 2.0
        spikes[(input_norm > 0.5) & (input_norm <= 1.5)] = 1.0
        spikes[input_norm < -0.5] = -1.0
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        (input_norm,) = ctx.saved_tensors
        gamma = 1.5
        grad_neg = torch.clamp(1.0 - torch.abs(input_norm - (-0.5)) / gamma, min=0.0)
        grad_pos1 = torch.clamp(1.0 - torch.abs(input_norm - 0.5) / gamma, min=0.0)
        grad_pos2 = torch.clamp(1.0 - torch.abs(input_norm - 1.5) / gamma, min=0.0)
        grad_mask = SURROGATE_ALPHA * (grad_neg + grad_pos1 + grad_pos2)
        return grad_output * grad_mask


class QuaternaryLIF(nn.Module):
    def __init__(self, decay: float = 0.875):
        super().__init__()
        self.decay = decay
        self.spike_op = TrainableQuaternarySpikeAct.apply
        self.fire_rate = 0.0
        self.v_th = nn.Parameter(torch.tensor(2.0))
        self.batch_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
        self.inference_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

    def reset_statistics(self) -> None:
        self.inference_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

    def forward(self, x, mem):
        mem = self.decay * mem + x
        normalized_mem = mem / self.v_th
        raw_spike = self.spike_op(normalized_mem)
        with torch.no_grad():
            self.fire_rate = (raw_spike != 0).float().mean().item()
            for value in (-1, 0, 1, 2):
                count = int((raw_spike == float(value)).sum().item())
                self.batch_spike_counts[value] = count
                self.inference_spike_counts[value] += count
        spike = raw_spike * self.v_th
        reset_val = raw_spike * self.v_th
        mem = mem - reset_val
        return spike, mem
