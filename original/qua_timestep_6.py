import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# [新增] 用於繪製與計算混淆矩陣
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Hyper-parameters & Configuration
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.05  
WEIGHT_DECAY = 5e-4   
NUM_EPOCHS = 50
TIME_STEPS = 6

TAU = 0.9
ALPHA = 0.5   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Trainable Quaternary Spike Code (-1, 0, 1, 2)
# ==========================================
class TrainableQuaternarySpikeAct(torch.autograd.Function):
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
        input_norm, = ctx.saved_tensors
        # [改良 2] 放寬 gamma，讓梯度景觀更平滑，減少越界時的震盪
        gamma = 1.5 
        
        grad_neg  = torch.clamp(1.0 - torch.abs(input_norm - (-0.5)) / gamma, min=0.0)
        grad_pos1 = torch.clamp(1.0 - torch.abs(input_norm - 0.5) / gamma, min=0.0)
        grad_pos2 = torch.clamp(1.0 - torch.abs(input_norm - 1.5) / gamma, min=0.0)
        
        grad_mask = ALPHA * (grad_neg + grad_pos1 + grad_pos2)
        return grad_output * grad_mask

# ==========================================
# 修改 1: 拔除 self.a 的梯度，強制固定閾值，降低激發率
# ==========================================
# ==========================================
# 核心修改 1: 擴張並釋放閾值，強效壓制高激發率
# ==========================================
class QuaternaryLIF(nn.Module):
    def __init__(self, decay=TAU):
        super(QuaternaryLIF, self).__init__()
        self.decay = decay
        self.spike_op = TrainableQuaternarySpikeAct.apply
        self.fire_rate = 0.0
        # 將初始閾值拉高至 2.0，大幅拓寬 0 值的產生機率，恢復 SNN 的稀疏性
        self.v_th = nn.Parameter(torch.tensor(2.0)) 
        
        # [新增] 紀錄該 Batch 的各類 Spike 發生數量
        self.batch_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

    def forward(self, x, mem):
        mem = self.decay * mem + x
        # 除以 v_th 等同於縮小 input_norm，使其更容易落在 -0.5 ~ 0.5 的不發射區間
        normalized_mem = mem / self.v_th
        
        raw_spike = self.spike_op(normalized_mem)
        with torch.no_grad():
            self.fire_rate = (raw_spike != 0).float().mean().item()
            # [新增] 計算並儲存各 Spike 的數量
            self.batch_spike_counts[-1] = (raw_spike == -1.0).sum().item()
            self.batch_spike_counts[0]  = (raw_spike == 0.0).sum().item()
            self.batch_spike_counts[1]  = (raw_spike == 1.0).sum().item()
            self.batch_spike_counts[2]  = (raw_spike == 2.0).sum().item()
            
        spike = raw_spike * self.v_th
        reset_val = raw_spike * self.v_th
        mem = mem - reset_val
        return spike, mem

# ==========================================
# 3. Network Architectures 
# ==========================================
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=0.01) 
        self.lif1 = QuaternaryLIF() 
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.lif2 = QuaternaryLIF() 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.01)
            )

    def forward(self, x, mem_dict, prefix):
        out = self.bn1(self.conv1(x))
        mem_key1 = f'{prefix}_lif1'
        if mem_dict[mem_key1] is None:
            mem_dict[mem_key1] = torch.zeros_like(out)
        out_s1, mem_dict[mem_key1] = self.lif1(out, mem_dict[mem_key1])
        
        out_conv2 = self.bn2(self.conv2(out_s1))
        shortcut_out = self.shortcut(x)
        total_input = out_conv2 + shortcut_out
        
        mem_key2 = f'{prefix}_lif2'
        if mem_dict[mem_key2] is None:
            mem_dict[mem_key2] = torch.zeros_like(total_input)
        out_s2, mem_dict[mem_key2] = self.lif2(total_input, mem_dict[mem_key2])
        return out_s2, mem_dict

class SResNet20(nn.Module):
    def __init__(self):
        super(SResNet20, self).__init__()
        self.in_planes = 32
        self.conv1 = conv3x3(3, 32)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
        self.lif1 = QuaternaryLIF()
        self.layer1 = self._make_layer(32, 3, stride=1)
        self.layer2 = self._make_layer(64, 3, stride=2)
        self.layer3 = self._make_layer(128, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # [改良 3] 加入 Dropout 抑制過擬合
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(128, 10)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.ModuleList(layers)

    def forward(self, x):
        mem_dict = {}
        mem_dict['input_lif'] = None 
        for i in range(3):
            for b in range(3):
                mem_dict[f'l{i}_b{b}_lif1'] = None
                mem_dict[f'l{i}_b{b}_lif2'] = None
        spike_sum = 0
        for t in range(TIME_STEPS):
            out = self.bn1(self.conv1(x))
            if mem_dict['input_lif'] is None:
                mem_dict['input_lif'] = torch.zeros_like(out)
            out, mem_dict['input_lif'] = self.lif1(out, mem_dict['input_lif'])
            for i, block in enumerate(self.layer1):
                out, mem_dict = block(out, mem_dict, f'l0_b{i}')
            for i, block in enumerate(self.layer2):
                out, mem_dict = block(out, mem_dict, f'l1_b{i}')
            for i, block in enumerate(self.layer3):
                out, mem_dict = block(out, mem_dict, f'l2_b{i}')
            
            out_pool = self.avg_pool(out)
            out_pool = out_pool.view(out_pool.size(0), -1)
            
            # 在最終分類前套用 Dropout
            out_pool = self.dropout(out_pool)
            spike_sum += self.fc(out_pool)
            
        return spike_sum / TIME_STEPS

# ==========================================
# 4. Data & Training Utils
# ==========================================
def get_dataloader():
    print("Preparing Data (Full CIFAR10)...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# ==========================================
# 核心修改 2: 訓練迴圈 (配合 OneCycleLR 的 Batch 更新)
# ==========================================
def train_one_epoch(model, trainloader, optimizer, criterion, scheduler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    avg_fire_rate = 0.0
    
    # [新增] 統計整個 Epoch 內各種類 Spike 的發生次數
    epoch_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # [注意] OneCycleLR 必須在每個 Batch 這裡進行 step
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        fire_rates = []
        for m in model.modules():
            if isinstance(m, QuaternaryLIF):
                fire_rates.append(m.fire_rate)
                # [新增] 把這個 batch 記錄到的數量累加上去
                epoch_spike_counts[-1] += m.batch_spike_counts[-1]
                epoch_spike_counts[0]  += m.batch_spike_counts[0]
                epoch_spike_counts[1]  += m.batch_spike_counts[1]
                epoch_spike_counts[2]  += m.batch_spike_counts[2]

        current_avg_fire = sum(fire_rates) / len(fire_rates) if fire_rates else 0.0
        avg_fire_rate += current_avg_fire
        
        if (i+1) % 200 == 0:
            print(f'[Quaternary Model] Step [{i+1}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.1f}%, LR: {scheduler.get_last_lr()[0]:.5f}')
            
    epoch_acc = 100. * correct / total
    epoch_avg_fire = (avg_fire_rate / len(trainloader)) * 100
    print(f'[Quaternary Model] Epoch {epoch+1} Time: {time.time()-start_time:.2f}s, Final Train Acc: {epoch_acc:.2f}%, Avg Fire: {epoch_avg_fire:.2f}%')
    # [修改] 一併回傳 Spike 的數量字典
    return epoch_acc, epoch_avg_fire, epoch_spike_counts

# [修改] 新增 return_cm 參數，如果為 True 就一併回傳標籤與預測結果做混淆矩陣
def test(model, testloader, return_cm=False):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if return_cm:
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

    acc = 100.*correct/total
    if return_cm:
        return acc, all_targets, all_preds
    return acc

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == '__main__':
    NUM_EPOCHS = 100
    # OneCycleLR 的最高峰值學習率
    MAX_LR = 3e-3 
    
    trainloader, testloader = get_dataloader()
    
    print(f"Initializing SResNet20 (Quaternary) on {DEVICE}...")
    model = SResNet20().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 使用 AdamW 搭配微調過的 Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=5e-4)
    
    # 導入 OneCycleLR 強力打破準確率停滯
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(trainloader), 
        epochs=NUM_EPOCHS,
        pct_start=0.15 # 前 15% Epoch 進行預熱爬升
    )
    
    results_data = []
    
    # [新增] 用來記錄最佳測試準確率與其對應的預測結果
    best_test_acc = 0.0
    best_targets = []
    best_preds = []

    print("Start Quaternary Training with OneCycleLR & V_th Expansion...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_acc, fire_rate, spike_counts = train_one_epoch(model, trainloader, optimizer, criterion, scheduler, epoch)
        
        # [修改] 每個 Epoch 都獲取預測結果，方便我們隨時擷取「最佳」的那一次
        test_acc, all_targets, all_preds = test(model, testloader, return_cm=True)
        
        print(f"Epoch {epoch+1} Result: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # [新增] 只要目前的 Test Acc 超越歷史最佳紀錄，就儲存權重與預測結果
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_targets = all_targets
            best_preds = all_preds
            print(f"🌟 [New Best] Test Acc improved to {best_test_acc:.2f}%! Saving best weights...")
            torch.save(model.state_dict(), "quaternary_sresnet20_v9_best_weights.pth")
        
        results_data.append({
            "Epoch": epoch + 1,
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Avg Fire Rate": fire_rate,
            "Spike -1": spike_counts[-1],
            "Spike 0": spike_counts[0],
            "Spike 1": spike_counts[1],
            "Spike 2": spike_counts[2]
        })

    # [修改] 使用我們紀錄到的「最佳 Epoch」預測結果來畫混淆矩陣
    print(f"Saving Best Confusion Matrix (Acc: {best_test_acc:.2f}%) to confusion_matrix_v9.png...")
    cm = confusion_matrix(best_targets, best_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Best Test Acc: {best_test_acc:.2f}%)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig('confusion_matrix_v9.png')
    plt.close()

    # Save to CSV
    print("Saving results to quaternary_training_log_v9.csv...")
    df = pd.DataFrame(results_data)
    df.to_csv("quaternary_training_log_v9.csv", index=False)
    print("Saved quaternary_training_log_v9.csv")

    # Plotting
    print("Plotting quaternary_result_v9.png...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 繪製準確度曲線 (左側 Y 軸)
    line1 = ax1.plot(df["Epoch"], df["Train Acc"], label='Train Acc', linestyle='--', color='blue', alpha=0.7)
    line2 = ax1.plot(df["Epoch"], df["Test Acc"], label='Test Acc', color='orange', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 繪製 Spike 發生數量曲線 (右側 Y 軸)
    ax2 = ax1.twinx()
    line3 = ax2.plot(df["Epoch"], df["Spike -1"], label='Count: Spike -1', color='red', linestyle=':')
    line4 = ax2.plot(df["Epoch"], df["Spike 0"], label='Count: Spike 0', color='grey', linestyle=':')
    line5 = ax2.plot(df["Epoch"], df["Spike 1"], label='Count: Spike 1', color='green', linestyle=':')
    line6 = ax2.plot(df["Epoch"], df["Spike 2"], label='Count: Spike 2', color='purple', linestyle=':')
    ax2.set_ylabel('Spike Count (Scientific)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    
    # 合併兩個軸的圖例
    lines = line1 + line2 + line3 + line4 + line5 + line6
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.95, 0.25))

    info_text = (
        f'Best Test Acc: {best_test_acc:.2f}%\n'  # [修改] 顯示最佳 Test Acc
        f'Final Train Acc: {df["Train Acc"].iloc[-1]:.2f}%\n'
        f'Final Fire Rate: {df["Avg Fire Rate"].iloc[-1]:.2f}%'
    )
    
    ax1.text(0.95, 0.05, info_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f'SResNet20 Quaternary Stable Version v9 ({TIME_STEPS} Steps)')
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig('quaternary_result_v9.png')
    print("Saved quaternary_result_v9.png")