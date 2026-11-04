# 300症例から296症例の解析を許可する統合ランナー

### 結論

今回の4症例では、従来DTW入力の6コード（20039/20011/20101/20742/20743/20108）が観測されていません。元データが空ではありません。欠損を0に置き換えて300症例とみなすことはしません。

この版は既存のDTW/Ablation/Two-stageの解析アルゴリズムを変更せず、296/300症例で解析を継続する**明示的オプション**を追加します。`--allow-case-shortfall` がない従来の挙動は変わりません。

## Ubuntuコマンド

```bash
source ~/venvs/chair_c31/bin/activate
cd /media/knaito/Samsung_SSD/C31/chair_300_pipeline_shortfall_v2
python run_300_pipeline.py --output-root /media/knaito/Samsung_SSD/C31/Output --stage all --resume --allow-case-shortfall --min-case-fraction 0.95
```

出力は以前と同じ `Output/chair_dtw_v3_1_300_casefirst`、`chair_ablation_300`、`chair_two_stage_300` を使用します。旧ファイルは必要に応じて再計算・上書きされます。過去の分析を保存したい場合は `--dtw-out` 等で別の出力名を指定してください。

**期待値:** DTWは296症例／300 Activity、AblationとTwo-stageは同一296症例／300 Activity。`case_coverage_summary.json`、`case_coverage.csv`、`cluster_assignments.csv`で確認してください。もし300 Activity未満なら品質チェックで停止します。

**注意:** コードの変更は*有効症例数の検証と次段階への伝播のみ*です。元Hybrid区間・DTW距離式・Ablation入力特徴・Two-stage計算法には変更ありません。300症例すべての分析とは表示しません。将来300症例を必須にする場合、Hybridの症例母集団を追加拡張し、6コードが記録されている有効症例から300を選ぶか、全症例に共通するイベント特徴設計に変更して再検証してください。
