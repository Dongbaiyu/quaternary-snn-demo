# Streamlit Web 部署指南

此 Web 版共用既有 `qsn_model`、`Predictor`、四組已訓練 checkpoint、metadata、Spike 統計與 `results/`，不修改 PySide6 GUI、原始模型程式或權重。

## 本機預覽

使用 Python 3.11：

```bash
cd /path/to/quaternary-snn-demo
python -m pip install -r requirements-web.txt
streamlit run web_app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

瀏覽器開啟 `http://localhost:8501`。VSCode Remote 使用者可在 Ports 面板轉送 8501。

Web Predictor 固定以 CPU 載入 checkpoint；程式不下載 CIFAR-10。十張展示圖片已放在 `web_app/assets/demo_samples/`，每個 CIFAR-10 類別各一張。

## 專用 Time-step 模型

推論頁提供 T=1、2、4、6。每個選項會從 `inference/model_registry.py` 載入不可拆分的 checkpoint、training T、TAU 與來源。模型資訊頁同時顯示 `results/all_time_step_results/` 的原始實驗紀錄及目前 checkpoint 的完整重評結果：

| T | TAU | Recorded Train | Recorded Best Test | Checkpoint Re-evaluation | Correct / 10000 | Source |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.875 | 98.96% | 91.18% | 90.99% | 9099 | `qua_timestep_1.py` |
| 2 | 0.875 | 99.57% | 92.04% | 91.93% | 9193 | `qua_timestep_2.py` |
| 4 | 0.875 | 99.84% | 92.03% | 92.03% | 9203 | `qua_v10.py` |
| 6 | 0.9 | 99.84% | 92.30% | 92.07% | 9207 | `qua_timestep_6.py` |

Recorded 數值來自使用者保存的訓練成果圖；Checkpoint Re-evaluation 是用目前 registry 權重重新跑 CIFAR-10 test set 10,000 張所得。除 T=4 外兩者並不完全相同，因此 UI 不會把兩者合併成同一種準確率。

`st.cache_resource` 的模型身分包含 checkpoint path、SHA-256、training T 與 TAU，因此 T=6 不會沿用 T=4 模型。切換模型會清除上一個模型的推論結果，重新推論後才顯示 Top-1、Top-3、Softmax、耗時與 Spike 統計。

## Streamlit Community Cloud

### 推論在哪裡執行

正式網站不會連回開發電腦或實驗主機。Streamlit Community Cloud
會從 GitHub 取得程式、checkpoint、metadata 與展示圖片，再由雲端容器的
CPU 完成真實推論。模型由 `st.cache_resource` 依 checkpoint、SHA-256、Time-step
與 TAU 分別快取，不需要 VPN、SSH 或對本機的網路存取。

`demo_assets/` 中的截圖與匯出結果只是展示備援及驗證證據，網站的
「開始推論」仍會執行 Predictor 與對應權重，不會把預先寫好的結果冒充即時推論。

### 比賽當天建議

1. 上台前 10–15 分鐘開啟公開網址，避免 Community Cloud 休眠後的冷啟動等待。
2. 依序選擇 T=1、2、4、6 並各執行一次，讓四個 Predictor 進入快取。
3. 報告電腦同時保留本專案與可執行的本機 Python 環境；雲端服務異常時，使用
   `bash scripts/launch_demo.sh` 或本機 Streamlit，不需要連回開發主機。
4. `demo_assets/screenshots/` 保留正式畫面，作為最後一層靜態備援。

部署所需檔案已放入 repository：

- Entrypoint：`web_app/streamlit_app.py`
- App 優先依賴：`web_app/requirements.txt`
- 根目錄副本：`requirements-web.txt`
- 設定：`.streamlit/config.toml`
- Python 開發版本提示：`.python-version`
- 模型 registry：`inference/model_registry.py`
- 模型：`checkpoints/qsn_sresnet20_t4_best.pth` 與 `original/` 下 T=1、T=2、T=6 權重及同名 JSON
- 展示資產：`web_app/assets/demo_samples/`
- 成果圖片：`results/`

Community Cloud 會先搜尋 entrypoint 所在目錄的 dependency file，因此 `web_app/requirements.txt` 會優先於根目錄既有的 `environment.yml`，避免安裝 CUDA lock。

部署步驟：

1. 將目前 repository push 至 GitHub；不要提交密碼、Token 或 `.streamlit/secrets.toml`。
2. 登入 [Streamlit Community Cloud](https://share.streamlit.io/) 並選擇 Create app。
3. 選擇 repository 與 branch。
4. Main file path 填入 `web_app/streamlit_app.py`。
5. 在 Advanced settings 將 Python version 選為 **3.11**。Community Cloud 的 Python 版本由部署介面選擇，建立後若要更換需重新部署。
6. 不需要設定 secrets；完成後按 Deploy。

正式展示可在公開網址後加上 `?embed=true`，隱藏 Streamlit
toolbar 與外框，例如 `https://<app-name>.streamlit.app/?embed=true`。
`Manage app` 是 Community Cloud 提供給已登入且具有管理權限使用者的控制項，
不是本網站的功能，也不會將管理權限交給一般訪客。

公開環境的 `.streamlit/config.toml` 已關閉瀏覽器端完整錯誤堆疊與
開發者選單。完整 log 只由 Community Cloud 管理介面檢視；
repository 不需要、也不應設定任何 secrets。

官方文件：

- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Deploy and choose Python version](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)

## CPU 與資源

`requirements-web.txt` 使用 PyTorch 2.10.0 / torchvision 0.25.0 CPU wheels，不沿用 `cu128` lock。可重跑量測：

```bash
python scripts/benchmark_web_cpu.py
```

量測結果會寫入 `results/web_cpu_benchmark.json`。本機環境使用 CUDA build 的 PyTorch 但 Web service 強制 `device=cpu`；Community Cloud 安裝的是 CPU wheel，因此雲端實際 RSS 可能更低。

## 多使用者行為

- `st.cache_resource` 在一個 server process 中為每組 registry 身分只建立一份模型。
- `threading.Lock` 保護整段 Predictor inference 與 Spike collector。
- 每位使用者的歷史只存於自己的 `st.session_state`。
- CSV 透過 `st.download_button` 在記憶體產生，不寫入 server 共用檔案。
