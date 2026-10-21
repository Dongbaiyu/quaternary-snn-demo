# 四元 SNN 推論與視覺化 Demo

這是以 `original/qua_v10.py` 為唯一四元 SNN 軟體基準的 CIFAR-10 SResNet20 重構版。原始檔案保留不變；新系統將模型、資料、訓練、評估、推論與 PySide6 GUI 分離。詳細分析見 `ANALYSIS.md`。

## 安裝

建議在專用 virtualenv/Conda environment 內執行：

```bash
cd /path/to/quaternary-snn-demo
python -m pip install -r requirements.txt
```

Demo 不會下載資料。它會先找 `創新營/data`，若不存在則使用目前已有的 `../Demo/data`。

## 正式競賽啟動

正式展示建議使用專用 Conda 環境：

```bash
conda env create -f environment.yml
conda activate qsn-demo
cd /path/to/quaternary-snn-demo
bash scripts/launch_demo.sh
```

啟動腳本只執行環境、CIFAR-10、checkpoint、metadata、results、展示案例及 parity 檢查；不會安裝套件、下載資料、訓練或量化。單獨執行檢查：

```bash
python scripts/preflight_check.py
```

已驗證版本記錄於 `environment.yml` 與 `requirements-lock.txt`。

正式 checkpoint：

- 路徑：`checkpoints/qsn_sresnet20_t4_best.pth`
- 大小：4,404,231 bytes
- SHA-256：`5203decd9d22ef22db755a37f1b3329699fd20a0f1e7823c6316421835c2d31f`


## 執行 Demo

```bash
python gui/main.py
```

GUI 啟動時掃描 `checkpoints/*.pth`，只顯示同名 JSON metadata 存在且欄位合法的 checkpoint。目前只提供經訓練紀錄驗證的 T=4 FP32 模型（Test Accuracy 92.03%）。推論在 `QThread` 內執行，不會執行 backward、optimizer、scheduler、訓練或 INT8 量化。

競賽展示版包含：

- 「系統總覽」首頁：真實 checkpoint 資訊、執行裝置、SNN 流程與一鍵快速展示。
- 「推論」頁：CIFAR-10 圖片、模型設定、Top-1/Top-3 softmax probability、十類分數圖與實際四元脈衝分布。
- 「模型資訊」頁：從既有 metadata、CSV 與 PNG 顯示 92.03% 成果、訓練曲線及 confusion matrix。
- 「執行紀錄」頁：研究模式下顯示，並可匯出 CSV。

「展示模式」只顯示總覽、推論與模型資訊；「研究模式」顯示全部四頁。這個切換只影響 UI。快速展示預設 image index、模式、結果目錄與視窗大小來自 `configs/default.json`，不會修改模型參數。

遠端環境可用 offscreen 模式檢查啟動：

```bash
QT_QPA_PLATFORM=offscreen python gui/main.py
```

## 評估

```bash
python training/evaluate.py --checkpoint checkpoints/qsn_sresnet20_t4_best.pth
```

評估只使用本機現有 CIFAR-10 test set，不自動下載。

## 訓練

```bash
python training/train.py --time-steps 4
```

這是離線訓練入口，不會被 GUI 呼叫。它保持原始 AdamW、OneCycleLR、100 epochs、label smoothing 與 gradient clipping 行為，並在最佳權重旁寫入 metadata。若資料已存在且不允許下載，加上 `--no-download`。

## 驗證重構一致性

```bash
python -m pytest tests/test_parity.py -q
python -m pytest tests/test_gui.py -q
```

一致性測試將原始與重構後 SResNet20 載入同一份 checkpoint，在 `eval()` / `no_grad()` 下比對相同輸入的 logits，並驗證 GUI 掃描只暴露 T=4。GUI 測試以 offscreen 模式驗證四頁初始化、QThread 推論、結果圖表、無 checkpoint、缺少 data/results、worker 關閉安全與 1366×768／1920×1080 版面。脈衝測試另驗證 reset、20 次連續 inference、hook、logits 與記憶體穩定性。

完整交付驗證：

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
python scripts/preflight_check.py
python training/evaluate.py --checkpoint checkpoints/qsn_sresnet20_t4_best.pth
```

## 固定展示案例與備援資產

`configs/demo_samples.json` 保存十個經正式 T=4 Predictor 重新驗證的正確案例，完整涵蓋 CIFAR-10 十個類別，每類各一張。首頁快速展示預設使用第一筆案例，推論頁也可從「展示案例」下拉選單切換。

離線備援位於：

- `demo_assets/screenshots/`：總覽、正確推論、實際脈衝統計與模型成果截圖。
- `demo_assets/exported_results/`：正式 checkpoint 產生的展示 inference JSON。

可重新產生：

```bash
QT_QPA_PLATFORM=offscreen python scripts/capture_demo_assets.py
```

## 研究聲明

本模型依據相關三元脈衝神經網路文獻之方法描述自行實作。由於原文未提供官方程式碼，本系統使用相同研究概念與本研究建立的訓練流程進行三元及四元模型比較，不宣稱完全復現原論文程式碼。

本階段未建立三元模型。後續若要建立，GUI 標籤將使用 `Self-implemented Ternary Baseline`，並會先提交待確認的公式與程式修改位置。
