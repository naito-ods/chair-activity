# 実行方法と補助スクリプトの配置先

このフォルダには、既存DTW割当を保存したまま実行する、非DTWの別バージョン解析を置いています。各フォルダの `RUN_ANALYSIS.txt` に再実行手順と入力条件があります。

- `chair_numeric_multiview_v1/`: 数値水準・イベントの複数ビュー比較と、記録方式を考慮した再評価。
- `chair_hp_sequence_v3/`: HP種類、回転領域、HP3後の低回転、高回転後の短い中回転の順序解析。
- `chair_hp2_similarity_v4/`: 高回転主体区間の作動構造とトルク・ペダル2による類似性比較。
- `chair_context_v5/`: 高回転を観測する症例全体でHP3と中・低回転の前後文脈を比較。97症例の全体図、重複候補、文脈クラスタ内部安定性、数値水準の補助比較を出力。

## リポジトリの構成

```text
chair-activity/
├── AGENTS.md
├── README.md
├── CLI_START_HERE.txt
├── chair_300_pipeline_shortfall_v2/
├── chair_two_stage_replay_v1/
├── scripts/
│   ├── README.md
│   ├── chair_numeric_multiview_v1/
│   ├── chair_hp_sequence_v3/
│   ├── chair_hp2_similarity_v4/
│   └── chair_context_v5/
└── .gitignore
```

元の C31 内のファイル・フォルダは保存しています。Pythonキャッシュ、症例データ、個別図、解析結果、仮想環境はGitHubに含めません。各解析は既存venvを使用し、狭い列の読み込みと中間表の再利用でメモリ使用を抑えます。

## 環境準備

リポジトリのルートから実行します。Python 3.10 を使用します。

```bash
source ~/venvs/chair_c31/bin/activate
python -m pip install -r chair_300_pipeline_shortfall_v2/requirements.txt
```

ルートや各フォルダの既存文書にある `C31/chair_...` のパスはコピー元の配置です。このリポジトリでは `C31/chair-activity/chair_...` に読み替えてください。データと既存解析結果は引き続きリポジトリ外の `C31/Output/` を参照します。

## 入力確認のみ

```bash
python chair_300_pipeline_shortfall_v2/run_300_pipeline.py --help
python chair_two_stage_replay_v1/visualize_two_stage_classes.py --help
```

## 新しい出力先を使用する296症例パイロットの例

解析を開始する際に使用する例です。既存結果を保存するため、出力先は未使用のディレクトリ名を指定してください。

```bash
python chair_300_pipeline_shortfall_v2/run_300_pipeline.py \
  --output-root /media/knaito/Samsung_SSD/C31/Output/chair_activity_new_run \
  --data-root /media/knaito/Samsung_SSD/C31/Output/chair_model_A_results \
  --hierarchy-root /media/knaito/Samsung_SSD/C31/Output/chair_hierarchical_v3_300 \
  --stage all --allow-case-shortfall --min-case-fraction 0.95
```

296/300症例の解析であり、300症例すべてのDTW解析完了とは表現しません。必要な前処理データとHybrid v3の結果は別途必要です。

## 既存結果のReplay

```bash
python chair_two_stage_replay_v1/visualize_two_stage_classes.py \
  --two-stage-root /media/knaito/Samsung_SSD/C31/Output/chair_two_stage_300 \
  --data-root /media/knaito/Samsung_SSD/C31/Output/chair_model_A_results \
  --out /media/knaito/Samsung_SSD/C31/Output/chair_activity_replay_new \
  --max-members 7 --max-points 500 --dpi 110
```

Replayの出力先も未使用のディレクトリ名を指定してください。解析データと生成レポートはローカルの `C31/Output/` に保存します。
