# Chair log 300症例パイプライン（Hybrid v3 → DTW → Ablation → Two-stage）

本パッケージは **既存のHybrid v3の300症例結果を読み取り、段階をつなげる**ものです。元DAT、前処理済みの `chair_model_A_results`、およびHybrid v3の分割は変更しません。DTW/特徴量除去/2段階モデルの既存計算コードを同梱しています。

## 既定のWindowsフォルダ

- 入力（前処理済み）：`C:\131share\DB\C31\Output\chair_model_A_results`
- 入力（300症例のHybrid）：`C:\131share\DB\C31\Output\chair_hierarchical_v3_300`
- DTW出力：`C:\131share\DB\C31\Output\chair_dtw_v3_1_300_casefirst`
- Ablation出力：`C:\131share\DB\C31\Output\chair_ablation_300`
- Two-stage出力：`C:\131share\DB\C31\Output\chair_two_stage_300`

Hybrid v3のフォルダ名が異なる場合は `--hierarchy-root "実際のパス"` を指定してください。

## 重要：症例300とActivity 300は別

初回は `--cases 300 --activities 300 --max-per-case 2 --max-bins 64`。症例優先サンプリングを採用し、**各症例が最低1個の使用可能Activityを提供することをチェックします**。1件も使用可能Activityがない症例があれば、勝手に症例数を減らしてクラスタリングを継続せずエラー停止。`case_coverage.csv` と `processing_errors.csv` で原因を確認します。全症例が含まれた後、必要に応じ `--activities 360 --max-per-case 2` に増やして再解析してください。**別設定は別出力フォルダに保存してください**。

初回300 Activityでの総ペア数は44,850、360 Activityなら64,620です。Ablationは複数のDTW条件を計算するため、CPUの計算負荷があります。Xeon E-2224G / 16 GB RAMでは初回は300 Activityを推奨します。`--max-bins 64` は以前の100症例DTWと合わせた値です。計算時間を優先する予備試験では `--max-bins 48` を選べますが、100症例とのSilhouetteや距離の比較では系列解像度の違いに注意してください。

## 実行（既存のvenvを使用）

ZIPを `C:\131share\DB\C31\chair_300_pipeline` に展開します。既存のvenv `C:\131share\DB\C31\.venv` を利用します。

```cmd
cd /d C:\131share\DB\C31\chair_300_pipeline
C:\131share\DB\C31\.venv\Scripts\python.exe -m pip install -r requirements.txt
C:\131share\DB\C31\.venv\Scripts\python.exe -m pip install numba
C:\131share\DB\C31\.venv\Scripts\python.exe test_logic.py
C:\131share\DB\C31\.venv\Scripts\python.exe test_case_first.py
C:\131share\DB\C31\.venv\Scripts\python.exe test_ablation.py
C:\131share\DB\C31\.venv\Scripts\python.exe test_two_stage.py
```

`numba` は任意ですが、AblationのDTWを高速化します。パッケージの実行には `requirements.txt` が必須です。

```cmd
C:\131share\DB\C31\.venv\Scripts\python.exe run_300_pipeline.py --stage all --resume
```

または `run_300.cmd` を実行します。`--resume` は、保存済み出力の症例数・Activity数・距離の一致を検査し、正常完了したステージを再実行しません。異なる設定での再解析は、`--dtw-out`、`--ablation-out`、`--two-stage-out` で出力先を変更してください。

### 段階ごとに実行・再開

```cmd
python run_300_pipeline.py --stage dtw
python run_300_pipeline.py --stage ablation
python run_300_pipeline.py --stage two-stage
python run_300_pipeline.py --stage check
```

`python` は `.venv` 有効化後の例です。終了後に元症例数やActivityが欠落していれば次のステージへは進めません。

### フォルダ名が違う場合

```cmd
python run_300_pipeline.py --hierarchy-root "C:\131share\DB\C31\Output\ここに実際のHybrid300フォルダ名" --stage all --resume
```

### 16GB RAMでの注意

- DTWは `--plot-mode diagnostics --no-exact-events`、Ablation/Two-stageはMedoid PNGの出力数を0にしており、**解析優先・省メモリ構成**です。DTW結果ができた後に必要な代表図だけ別途描画してください。
- ペアワイズDTW自体は主にCPUで処理。GPUは不要です。
- 同じPCでメモリを大量に使うアプリを閉じて実行してください。
- 読込中に再度メモリ不足が起きれば、個々の症例の巨大CSVが原因の可能性があります。`--resume` だけでは未完了DTW内部からの続行はできないため、当該症例の特定・読込の最適化が必要です。
- 300例を選べるかどうかは `chair_hierarchical_v3_300` の実際の `case_id` と `chair_model_A_results/cases/` の対応次第です。

## 入力ファイル・検査

`--hierarchy-root` に以下が必要です。

- `hierarchical_hybrid_activity_intervals.csv`
- `level1_operations.csv`
- `level1_episodes.csv`
- `activity_intervals_all_methods.csv`

DTW後に `case_coverage_summary.json` / `case_coverage.csv` / `cluster_assignments.csv` / `dtw_distance_matrix.csv` を照合します。Ablation後に `fixed_activity_manifest.csv` で**同じActivityのID・時刻**であることを検査し、`baseline_fidelity.json` のA対ベースライン距離誤差が 1e-5以下であることをチェックします。Two-stage後は `stage1_fixed_assignments.csv` の300症例が維持されていることを確認します。数値的に適格なActivityが少ない場合、Stage2は `NOT_CLUSTERED` となっても正常な研究結果です。

出力先3フォルダは、従来100症例の結果を上書きしません。

## 解釈の制限

300症例から選んだ300 Activityは**各症例を均等に代表する設計**であり、全診療操作の出現頻度を再現した標本ではありません。解析入力の状態コードは `20039,20011,20101`、数値コードは `20742,20743,20108` の既定値を用い、全イベントコードを直接距離計算するものではありません。Stage1のHP1/HP2等は暫定ラベルであり臨床処置名の教師なし発見が証明されたわけではありません。

## 追加：Activity数を360にする例

```cmd
python run_300_pipeline.py --cases 300 --activities 360 --max-per-case 2 --max-bins 64 --dtw-out "C:\131share\DB\C31\Output\chair_dtw_v3_1_300_360act" --ablation-out "C:\131share\DB\C31\Output\chair_ablation_300_360act" --two-stage-out "C:\131share\DB\C31\Output\chair_two_stage_300_360act" --stage all --resume
```
