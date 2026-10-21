from __future__ import annotations

import torch
import torch.nn as nn

from .neurons import QuaternaryLIF

def normalize_time_steps(value) -> int:
    """Normalize integer and UI-style Time-step values without changing semantics."""
    if isinstance(value, bool):
        raise ValueError("time_steps must be a positive integer")
    if isinstance(value, str):
        candidate = value.strip()
        if candidate[:1].lower() == "t":
            candidate = candidate[1:].strip()
            if candidate.startswith("="):
                candidate = candidate[1:].strip()
        if not candidate.isdecimal():
            raise ValueError(f"invalid time_steps value: {value!r}")
        normalized = int(candidate)
    else:
        try:
            normalized = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid time_steps value: {value!r}") from exc
        try:
            if normalized != value:
                raise ValueError(f"invalid time_steps value: {value!r}")
        except TypeError as exc:
            raise ValueError(f"invalid time_steps value: {value!r}") from exc
    if normalized < 1:
        raise ValueError("time_steps must be at least 1")
    return normalized



def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, tau: float = 0.875):
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.lif1 = QuaternaryLIF(tau)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.lif2 = QuaternaryLIF(tau)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.01),
            )

    def forward(self, x, mem_dict, prefix):
        out = self.bn1(self.conv1(x))
        mem_key1 = f"{prefix}_lif1"
        if mem_dict[mem_key1] is None:
            mem_dict[mem_key1] = torch.zeros_like(out)
        out_s1, mem_dict[mem_key1] = self.lif1(out, mem_dict[mem_key1])
        out_conv2 = self.bn2(self.conv2(out_s1))
        total_input = out_conv2 + self.shortcut(x)
        mem_key2 = f"{prefix}_lif2"
        if mem_dict[mem_key2] is None:
            mem_dict[mem_key2] = torch.zeros_like(total_input)
        out_s2, mem_dict[mem_key2] = self.lif2(total_input, mem_dict[mem_key2])
        return out_s2, mem_dict


class SResNet20(nn.Module):
    """The qua_v10.py SResNet20 with time_steps and tau made configurable."""

    def __init__(self, time_steps: int = 4, tau: float = 0.875):
        super().__init__()
        self.time_steps = normalize_time_steps(time_steps)
        self.tau = float(tau)
        self.in_planes = 32
        self.conv1 = conv3x3(3, 32)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
        self.lif1 = QuaternaryLIF(tau)
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 3, stride=2)
        self.layer3 = self._make_layer(128, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 10)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, block_stride, self.tau))
            self.in_planes = planes
        return nn.ModuleList(layers)

    def reset_spike_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, QuaternaryLIF):
                module.reset_statistics()

    def spike_statistics(self) -> dict[int, int]:
        totals = {-1: 0, 0: 0, 1: 0, 2: 0}
        for module in self.modules():
            if isinstance(module, QuaternaryLIF):
                for value, count in module.inference_spike_counts.items():
                    totals[value] += count
        return totals

    def forward(self, x, time_steps=None):
        current_time_steps = self.time_steps if time_steps is None else normalize_time_steps(time_steps)
        mem_dict = {"input_lif": None}
        for i in range(3):
            for b in range(3):
                mem_dict[f"l{i}_b{b}_lif1"] = None
                mem_dict[f"l{i}_b{b}_lif2"] = None
        spike_sum = 0
        for _ in range(current_time_steps):
            out = self.bn1(self.conv1(x))
            if mem_dict["input_lif"] is None:
                mem_dict["input_lif"] = torch.zeros_like(out)
            out, mem_dict["input_lif"] = self.lif1(out, mem_dict["input_lif"])
            for i, block in enumerate(self.layer1):
                out, mem_dict = block(out, mem_dict, f"l0_b{i}")
            for i, block in enumerate(self.layer2):
                out, mem_dict = block(out, mem_dict, f"l1_b{i}")
            for i, block in enumerate(self.layer3):
                out, mem_dict = block(out, mem_dict, f"l2_b{i}")
            out_pool = self.avg_pool(out)
            out_pool = out_pool.view(out_pool.size(0), -1)
            out_pool = self.dropout(out_pool)
            spike_sum += self.fc(out_pool)
        return spike_sum / current_time_steps
