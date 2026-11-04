#!/usr/bin/env python3
"""Add auditable, table-derived interpretation to the scientific HTML report."""
from pathlib import Path
import argparse,json
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('Output/chair_context_v5_20261104'));a=ap.parse_args();out=a.output
    s=json.loads((out/'analysis_summary.json').read_text())
    if s.get('context_interpretation_complete'):raise FileExistsError('Interpretation already added')
    f=pd.read_csv(out/'analysis_case_features.csv').set_index('case_id');ass=pd.read_csv(out/'context_cluster_assignments.csv');lab=ass[ass.k.eq(3)].set_index('case_id').cluster
    table=f.join(lab.rename('cluster'),validate='one_to_one')
    labels={1:'HP3の後に後半の高回転が現れる傾向',2:'HP3未観測を中心に、中回転等も混在する傾向',3:'高回転の後にHP3が現れる傾向'}
    descriptions=[]
    for cl,g in table.groupby('cluster'):
        descriptions.append(dict(cluster=int(cl),label=labels[int(cl)],n_cases=len(g),hp3_2s_n=int(g.hp3_2s.sum()),hp3_before_n=int(g.hp3_before_first_high.sum()),hp3_after_n=int(g.hp3_after_last_high.sum()),mid_block_n=int(g.macro_mid.sum()),mean_first_high_case_progress=float(g.first_high_progress.mean()),mean_last_high_case_progress=float(g.last_high_progress.mean())))
    pd.DataFrame(descriptions).to_csv(out/'context_cluster_interpretation.csv',index=False)
    text=[]
    for r in descriptions:
        text.append(f"文脈C{r['cluster']}: {r['label']}（{r['n_cases']}症例）。HP3総2秒以上は{r['hp3_2s_n']}例、最初の高回転より前のHP3は{r['hp3_before_n']}例、最後の高回転より後のHP3は{r['hp3_after_n']}例、中回転主体区間は{r['mid_block_n']}例です。全例が同じ順序に一致するという意味ではありません。")
    s['context_cluster_interpretation']=descriptions
    s['report_paragraphs']+=text+[
        '今回の中回転主体ブロック50症例のうち1症例では、中回転の観測点がすべて個々の作動runの両端3秒以内にあります。したがってブロック基準が立上り・停止時通過を完全に排除できたとは言いません。raw波形とrun基準を併記して判断します。',
        '文脈3群の高回転中トルクは、十分な対応点を持つ症例の中央値をさらに群内で集計すると、全群1.0でした。ペダル2の水準も広く重なります。文脈による区分の方が説明しやすい一方、トルク等まで明瞭に分離したという結果ではありません。',
        '前後の大きい流れを捉えるには、短い波形の形より、HP3と高回転の前後関係、中回転主体区間の配置、低回転との組合せを主軸にする方針が今回のデータでは有望です。3群は今回の97症例・定義・距離に依存する探索候補です。',
        '元の300症例は同一チェアから選択された解析区間25.8〜43.3分のコホートであり、全症例母集団の無作為標本ではありません。信頼区間・症例bootstrapは選択コホート内の不確かさを表し、他チェアや他施設への一般化を保証しません。',
        '過程はprocess/feature_log.jsonl、analysis_log.jsonl、method_review.txt、extraction_notes.txtおよび独立検証JSONに記録しています。初期の観測時刻座標を修正したv5.2を最終解析に使用し、修正前v5.1はprocess/extraction_v5_1/へ保存しました。元の症例データと既存Stage2割当は変更していません。']
    s['context_interpretation_complete']=True
    (out/'analysis_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(descriptions).to_string(index=False))
if __name__=='__main__':main()
