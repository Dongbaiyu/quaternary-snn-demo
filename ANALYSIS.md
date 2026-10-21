# `qua_v10.py` 結構分析與重構方案

## 原始程式結構

`original/qua_v10.py` 共 495 行，將八種關注點放在同一檔案：

1. 全域訓練設定：Batch size、epoch、Time-step、TAU 與 device。
2. `TrainableQuaternarySpikeAct`：四元閾值及三個三角形 surrogate gradient。
3. `QuaternaryLIF`：可訓練 `v_th`、漏電累積、脈衝與 soft reset。
4. `BasicBlock` / `SResNet20`：32→64→128 通道的三階段 residual network。
5. CIFAR-10 訓練/測試前處理與 DataLoader。
6. 單一 epoch 訓練與完整測試迴圈。
7. 會就地修改模型的 INT8 + BNTT fusion 評估。
8. 訓練主程式、圖表、CSV 與 checkpoint 輸出。

## 重構原則

- `qsn_model/` 放數學行為、架構、設定與資料前處理。
- `training/` 封裝訓練與批次評估；GUI 不匯入訓練逻輯。
- `inference/Predictor` 是 GUI 唯一模型入口，固定使用 `eval()` 與 `no_grad()`。
- `checkpoints/*.json` 是可用模型白名單；權重或 metadata 任一缺失都不顯示。
- GUI 以 worker thread 執行載入與 forward，紀錄可匯出 CSV。
- INT8 破壞性轉換不進入一般 inference 路徑。

## 數學不變項目

四元輸出與邊界比較、surrogate gradient、`raw_spike * v_th`、soft reset、TAU=0.875、
BatchNorm momentum、SResNet20 層數/通道、dropout、Time-step 平均、CIFAR-10 normalization、
AdamW、OneCycleLR、gradient clipping 與 label smoothing 均保持原行為。
