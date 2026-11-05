# Repository Guidelines

## 專案結構與模組

本專案以 Isaac Lab 3.0 與 skrl PPO 訓練 Unitree G1，在崎嶇地形上學習步態。

- `src/Miislab_humanoid_gait_RL/`：Python 套件；`tasks/` 負責任務註冊，`assets/` 為資產模組入口。
- `tasks/humanoid_gait_rl_direct/config/g1/`（相對於套件）：環境、獎勵與 `agents/skrl_ppo_cfg.yaml` 設定。
- `tests/test_registration.py`：驗證任務與環境、代理程式入口。
- `scripts/list_envs.py`：列出環境的相容包裝程式。
- `logs/skrl/humanoid_gait_rl_g1/`：產生的模型、設定與 TensorBoard 紀錄，不納入版本控制。

## 安裝、開發與執行

依 README 準備 Linux、NVIDIA GPU，以及 `../../isaac/IsaacLab-3.0.0` 原始碼；使用 Python 3.12，在本專案根目錄執行：

```bash
uv sync                            # 安裝依賴至 .venv
uv build                           # 建置套件
uv run isaaclab list_envs           # 確認任務註冊
uv run pytest                      # 執行測試
uv run ruff check .                 # 靜態檢查
uv run ruff format --check .        # 檢查格式
uv run pre-commit run --all-files   # 完整提交檢查，部分 hook 會修改檔案
```

訓練冒煙驗證：

```bash
uv run isaaclab train --rl_library skrl --task Miislab-Humanoid-Gait-Rl-G1-Direct-v1 --num_envs 64 --max_iterations 10
```

使用 `uv run tensorboard --logdir logs/skrl` 檢視曲線。

## 程式風格與修改範圍

Python 使用四個空白縮排、120 字元行長；函式與變數採 `snake_case`，類別採 `PascalCase`。由 Ruff 管理格式與 import 排序，保留版權標頭；YAML 維持兩個空白縮排。pre-commit 另檢查拼字、檔案格式與超過 2000 KB 的新增檔案。

遵循 README 的競賽限制：訓練調整僅修改 `rewards.py` 與 `agents/skrl_ppo_cfg.yaml` 中 `TODO: EDIT BELOW` 至 `TODO: END` 的區塊。獎勵函式只讀取環境，每個回傳項目須有同名權重，張量形狀為 `(num_envs,)`。

## 測試規範

使用 pytest；檔名採 `test_*.py`，函式採 `test_*`。目前未設定覆蓋率門檻；新增測試依用途使用 `unit`、`integration`、`smoke` 或 `kitless` 標記。註冊測試可用 `uv run pytest tests/test_registration.py` 單獨執行。獎勵或 PPO 變更須另驗證短程訓練能完成並產生 checkpoint；回報執行環境、指令與結果，未執行項目須註明。

## Commit 與 Pull Request

目前歷史僅有 `init push`，尚無固定訊息格式。建議使用簡短、明確的動作描述，例如 `Adjust foot slide penalty`，每次提交聚焦單一目的。PR 請說明修改動機、參數差異、驗證結果與相關 issue；步態變更附訓練曲線或影片及執行設定。勿提交 `.venv/`、訓練紀錄或模型權重。

## 代理程式操作規範

使用繁體中文回覆。所有 PowerShell `Get-Content` 指令必須明確指定 `-Encoding UTF8`，例如 `Get-Content -Path README.md -Encoding UTF8`。
