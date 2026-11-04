# Two-stage分類クラスの再現図 v1

前回の `chair_300_pipeline_shortfall_v2` で生成された Two-stage結果を読み込み、各Stage 2サブクラスの「代表Activity（Medoid）」と「異なる症例から選んだ同じクラスのActivity」の波形を可視化します。**クラスタリングとActivity抽出のやり直しは行いません**。

## Ubuntuでの実行

```bash
source ~/venvs/chair_c31/bin/activate
cd "/media/knaito/Samsung_SSD/C31/chair_two_stage_replay_v1"
python visualize_two_stage_classes.py \
  --two-stage-root "/media/knaito/Samsung_SSD/C31/Output/chair_two_stage_300" \
  --data-root "/media/knaito/Samsung_SSD/C31/Output/chair_model_A_results" \
  --max-members 7 --max-points 500 --dpi 110
```

## 出力

`chair_two_stage_300/replay_class_examples/`:
- `BXX_SYY_medoid_raw_1s.png`: 該当サブクラスのMedoidにおける状態コード3種・数値コード3種。横軸はActivity開始からの秒数。
- `BXX_SYY_members_overlay.png`: 異なる症例を優先して最大7 Activityの数値信号を重ね合わせ。横軸は個々のActivityの進捗率（0–100%）、**DTWにより時間軸を歪めた波形ではありません**。
- `class_recurrence.csv`: 各サブクラスに含まれるActivity数・症例数・Stage2ケース再サンプリングARI（**親クラス単位**）。
- `plotted_activities.csv`: 描画に使った元の症例IDとActivity ID、時間範囲。
- `plot_errors.csv`: 読込・描画エラー。

`raw_1s.csv.gz` と `observation_mask_1s.csv.gz` を用い、観測がない区間は0とみなしません。観測された1秒値のみを描画し、イベント原記録のミリ秒時刻とは区別します。圧縮ファイルを対象列だけチャンクで読み、16 GB RAMでのメモリ負荷を軽減します。既存のMedoid PNGとは時間解像度が異なる場合があります。

既存の再現性指標は `chair_two_stage_300/stage1_case_subsampling.csv` と `chair_two_stage_300/stage2_case_subsampling.csv` にあり、ARI中央値が1に近いほど内部の分割結果が安定しています。しかし未知の症例に対する再現性（外部検証）は別途必要です。

Stage2が `NOT_CLUSTERED` となった親クラスにはサブクラス図が生成されません。`stage2_summary.csv` で原因をご確認ください。
