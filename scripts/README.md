# 実行方法と補助スクリプトの配置先

このフォルダは今後の補助スクリプト用です。現時点の実行コードは、元の構成を維持して各解析フォルダに置いています。

## リポジトリの構成

```text
chair-activity/
├── AGENTS.md
├── README.md
├── CLI_START_HERE.txt
├── chair_300_pipeline_shortfall_v2/
├── chair_two_stage_replay_v1/
├── scripts/
│   └── README.md
└── .gitignore
```

元の C31 内のファイル・フォルダはそのまま残してコピーしました。Pythonキャッシュ、データ、解析結果、仮想環境は含めていません。解析コードと既存の研究文書はコピー元と同一です。

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

Replayの出力先も未使用のディレクトリ名を指定してください。今回のリポジトリ構築では解析・Replayは実行していません。
