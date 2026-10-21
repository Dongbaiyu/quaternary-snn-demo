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
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Hyper-parameters & Configuration (超參數設定)
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.05  
WEIGHT_DECAY = 5e-4   
NUM_EPOCHS = 100

# [講解] 核心測試：TIME_STEPS 縮減為 4，Leakage 對齊硬體移位器為 0.875
TIME_STEPS = 4
TAU = 0.875

ALPHA = 0.5   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==========================================
# 2. Trainable Quaternary Spike Code (-1, 0, 1, 2)
# ==========================================
class TrainableQuaternarySpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_norm):
        ctx.save_for_backward(input_norm)
        spikes = torch.zeros_like(input_norm)
        
        # ========================================================
        # [講解] 四元 (-1, 0, 1, 2) 脈衝的撰寫位置在此：
        # 定義了神經元電壓在不同區間時，硬體要發射的 2-bit 訊號。
        spikes[input_norm > 1.5] = 2.0
        spikes[(input_norm > 0.5) & (input_norm <= 1.5)] = 1.0
        spikes[input_norm < -0.5] = -1.0
        # ========================================================
        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        input_norm, = ctx.saved_tensors
        gamma = 1.5 
        
        grad_neg  = torch.clamp(1.0 - torch.abs(input_norm - (-0.5)) / gamma, min=0.0)
        grad_pos1 = torch.clamp(1.0 - torch.abs(input_norm - 0.5) / gamma, min=0.0)
        grad_pos2 = torch.clamp(1.0 - torch.abs(input_norm - 1.5) / gamma, min=0.0)
        
        grad_mask = ALPHA * (grad_neg + grad_pos1 + grad_pos2)
        return grad_output * grad_mask

# ==========================================
# 3. LIF 神經元模型 (QuaternaryLIF)
# ==========================================
class QuaternaryLIF(nn.Module):
    def __init__(self, decay=TAU):
        super(QuaternaryLIF, self).__init__()
        self.decay = decay
        self.spike_op = TrainableQuaternarySpikeAct.apply
        self.fire_rate = 0.0
        self.v_th = nn.Parameter(torch.tensor(2.0)) 
        self.batch_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}

    def forward(self, x, mem):
        mem = self.decay * mem + x
        normalized_mem = mem / self.v_th
        
        raw_spike = self.spike_op(normalized_mem)
        with torch.no_grad():
            self.fire_rate = (raw_spike != 0).float().mean().item()
            self.batch_spike_counts[-1] = (raw_spike == -1.0).sum().item()
            self.batch_spike_counts[0]  = (raw_spike == 0.0).sum().item()
            self.batch_spike_counts[1]  = (raw_spike == 1.0).sum().item()
            self.batch_spike_counts[2]  = (raw_spike == 2.0).sum().item()
            
        spike = raw_spike * self.v_th
        reset_val = raw_spike * self.v_th
        mem = mem - reset_val
        return spike, mem

# ==========================================
# 4. 網路架構 (SResNet20)
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
            out_pool = self.dropout(out_pool)
            spike_sum += self.fc(out_pool)
            
        return spike_sum / TIME_STEPS

# ==========================================
# 5. 資料集處理 (Data Utils)
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
# 6. 訓練與測試邏輯
# ==========================================
def train_one_epoch(model, trainloader, optimizer, criterion, scheduler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    avg_fire_rate = 0.0
    
    epoch_spike_counts = {-1: 0, 0: 0, 1: 0, 2: 0}
    
    for i, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        fire_rates = []
        for m in model.modules():
            if isinstance(m, QuaternaryLIF):
                fire_rates.append(m.fire_rate)
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
    return epoch_acc, epoch_avg_fire, epoch_spike_counts

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
# 7. 硬體精準模擬：純 INT8 + BatchNorm 折疊 (BN Fusion)
# ==========================================
def apply_pure_int8_with_bntt_fusion(model):
    """
    [講解] 這個函式展示了硬體工程師的浪漫：「算子融合」。
    我們會：
    1. 把所有 Conv 權重強迫變成沒有小數點的純 INT8 整數 (-127~127)。
    2. 把本來應該乘回去的 Scale，偷偷塞給後面的 BatchNorm (bntt_gamma) 吸收掉！
    3. 這樣一來，權重是純的 INT8，但準確度不會崩盤。
    """
    print("\n[BN Fusion] Integrating Quantization Scale into BNTT Gamma/Beta...")
    
    prev_conv = None
    prev_conv_name = ""
    
    with torch.no_grad():
        # 按順序掃描網路裡的所有層
        for name, module in model.named_modules():
            
            # 如果看到卷積層，先把它記下來，等等跟下一個出現的 BatchNorm 配對
            if isinstance(module, nn.Conv2d):
                prev_conv = module
                prev_conv_name = name
                
            # 如果看到了 BatchNorm，而且前面剛好有一個卷積層，這就是一個「CU 管線 (MAC -> BNTT)」！
            elif isinstance(module, nn.BatchNorm2d) and prev_conv is not None:
                # ------------------------------------
                # 步驟 A: 將前面的卷積變成純 INT8
                # ------------------------------------
                weight = prev_conv.weight.data
                abs_max = weight.abs().max()
                if abs_max == 0:
                    prev_conv = None
                    continue
                    
                scale = abs_max / 127.0
                
                # 強制轉純整數，不再還原！(例如 120.0, -85.0)
                w_pure_int8 = torch.clamp(torch.round(weight / scale), -127, 127).float()
                prev_conv.weight.data = w_pure_int8
                
                # ------------------------------------
                # 步驟 B: 把 Scale 融合進這層的 BatchNorm (BNTT) 裡
                # ------------------------------------
                gamma = module.weight.data        # BNTT Gamma
                beta = module.bias.data           # BNTT Beta
                mean = module.running_mean        # 常態分佈的平均
                var = module.running_var          # 常態分佈的變異數
                eps = module.eps
                
                # 標準差 (Sigma)
                std = torch.sqrt(var + eps)
                
                # [印出資訊] 紀錄修改前的第一個通道的 Gamma 數值供你對比
                orig_gamma_0 = gamma[0].item()
                
                # [核心數學] 讓 BNTT 吸收 Scale
                # 由於輸入訊號被 INT8 放大 (放大了 1/scale 倍)，我們必須讓新的 Gamma 乘上 scale 縮小回來
                gamma_hw = (gamma / std) * scale
                beta_hw = beta - (gamma / std) * mean
                
                # [印出資訊] 紀錄修改後的新的 Gamma
                new_gamma_0 = gamma_hw[0].item()
                
                # 將 PyTorch 的 BatchNorm 變成等同於硬體 BNTT 的運算
                module.weight.data = gamma_hw
                module.bias.data = beta_hw
                # 將 mean 和 var 歸零/歸一，這樣 PyTorch 在算 BN 時，就會變成單純的 x * gamma_hw + beta_hw
                module.running_mean.data = torch.zeros_like(mean)
                module.running_var.data = torch.ones_like(var) - eps 
                
                print(f" - Conv: {prev_conv_name:<15} | Quantized to PURE INT8 (Scale: {scale.item():.5f})")
                print(f"   └─ BNTT Fusion -> Orig Gamma[0]: {orig_gamma_0:7.4f} | New Gamma[0]: {new_gamma_0:7.4f}")
                
                # 配對完成，清空紀錄準備找下一個
                prev_conv = None

            # 最後一層全連接層 (FC) 沒有配對的 BatchNorm
            elif isinstance(module, nn.Linear):
                weight = module.weight.data
                abs_max = weight.abs().max()
                if abs_max > 0:
                    scale = abs_max / 127.0
                    w_pure_int8 = torch.clamp(torch.round(weight / scale), -127, 127).float()
                    module.weight.data = w_pure_int8
                    # 分類層的絕對大小不影響 Argmax 找最大機率，所以不需要補償 Scale
                    print(f" - FC: {name:<17} | Quantized to PURE INT8 (Scale ignored for argmax)")

# ==========================================
# 8. Main Execution (主執行迴圈)
# ==========================================
if __name__ == '__main__':
    MAX_LR = 3e-3 
    
    trainloader, testloader = get_dataloader()
    
    print(f"Initializing SResNet20 (Quaternary) on {DEVICE}...")
    model = SResNet20().to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(trainloader), 
        epochs=NUM_EPOCHS,
        pct_start=0.15 
    )
    
    results_data = []
    best_test_acc = 0.0
    best_targets = []
    best_preds = []
    best_state_dict = None

    print(f"Start Quaternary Training (TIME_STEPS={TIME_STEPS}, TAU={TAU})...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_acc, fire_rate, spike_counts = train_one_epoch(model, trainloader, optimizer, criterion, scheduler, epoch)
        test_acc, all_targets, all_preds = test(model, testloader, return_cm=True)
        
        print(f"Epoch {epoch+1} Result: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_targets = all_targets
            best_preds = all_preds
            import copy
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"🌟 [New Best] Test Acc improved to {best_test_acc:.2f}%!")
        
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

    # ---------------------------------------------------------
    # 訓練結束：繪製結果
    # ---------------------------------------------------------
    print(f"\nSaving Best Confusion Matrix to confusion_matrix_v10.png...")
    cm = confusion_matrix(best_targets, best_preds)
    plt.figure(figsize=(12, 10)) 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES)
    plt.title(f"Confusion Matrix (Best Test Acc: {best_test_acc:.2f}%)")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.xticks(rotation=45) 
    plt.savefig('confusion_matrix_v10.png')
    plt.close()

    print("Saving results to quaternary_training_log_v10.csv...")
    df = pd.DataFrame(results_data)
    df.to_csv("quaternary_training_log_v10.csv", index=False)

    print("Plotting quaternary_result_v10.png...")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    line1 = ax1.plot(df["Epoch"], df["Train Acc"], label='Train Acc', linestyle='--', color='blue', alpha=0.7)
    line2 = ax1.plot(df["Epoch"], df["Test Acc"], label='Test Acc', color='orange', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    line3 = ax2.plot(df["Epoch"], df["Spike -1"], label='Count: Spike -1', color='red', linestyle=':')
    line4 = ax2.plot(df["Epoch"], df["Spike 0"], label='Count: Spike 0', color='grey', linestyle=':')
    line5 = ax2.plot(df["Epoch"], df["Spike 1"], label='Count: Spike 1', color='green', linestyle=':')
    line6 = ax2.plot(df["Epoch"], df["Spike 2"], label='Count: Spike 2', color='purple', linestyle=':')
    ax2.set_ylabel('Spike Count (Scientific)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    
    lines = line1 + line2 + line3 + line4 + line5 + line6
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.95, 0.25))

    info_text = (
        f'Best Test Acc: {best_test_acc:.2f}%\n'  
        f'Final Train Acc: {df["Train Acc"].iloc[-1]:.2f}%\n'
        f'Final Fire Rate: {df["Avg Fire Rate"].iloc[-1]:.2f}%'
    )
    ax1.text(0.95, 0.05, info_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f'SResNet20 Quaternary Stable Version v10 ({TIME_STEPS} Steps)')
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig('quaternary_result_v10.png')
    
    # ---------------------------------------------------------
    # 4. FP32 與 純 INT8 破壞性測試比較
    # ---------------------------------------------------------
    print("\n=================================================")
    print("   Starting Final Quantization Evaluation")
    print("=================================================")
    
    model.load_state_dict(best_state_dict)
    fp32_final_acc = test(model, testloader)
    print(f"[Baseline] Original FP32 Accuracy: {fp32_final_acc:.2f}%")
    
    # 執行 BN 融合與純 INT8 轉換
    apply_pure_int8_with_bntt_fusion(model)
    int8_fusion_acc = test(model, testloader)
    print(f"\n[Quantized] PURE INT8 + BNTT Fusion Accuracy: {int8_fusion_acc:.2f}%")
    
    print(f"\n=> Accuracy Drop (FP32 - PURE INT8 Fusion): {fp32_final_acc - int8_fusion_acc:.2f}%")
    print("=================================================")
    
    torch.save(best_state_dict, "quaternary_sresnet20_v10_best_weights.pth")
    print("Saved quaternary_sresnet20_v10_best_weights.pth")