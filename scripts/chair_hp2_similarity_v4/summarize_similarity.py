"""Evidence-based synthesis of separate similarity axes; not clinical taxonomy."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output.resolve();source=out.parent/'chair_hp_sequence_300_v3_20261104'
    meta=json.loads((out/'feature_metadata.json').read_text())
    for name,expected in meta['source_sha256'].items():assert sha(source/name)==expected,name
    models=pd.read_csv(out/'models.csv');assign=pd.read_csv(out/'assignments.csv')
    units=pd.read_csv(out/'units.csv');raw=pd.read_csv(out/'features.csv').set_index('unit_id')
    cf=pd.read_csv(out/'case_features.csv').set_index('case_id')
    neighbor=pd.read_csv(out/'neighbors.csv');ns=pd.read_csv(out/'neighbor_stability_summary.csv')
    assert len(units)==203 and units.case_id.nunique()==67 and len(cf)==67
    assert not assign.duplicated(['model_id','unit_id']).any()
    assert not neighbor.duplicated(['level','view','unit_id','rank']).any()
    assert (neighbor.case_id!=neighbor.neighbor_case_id).all()
    assert neighbor.groupby(['level','view','unit_id']).neighbor_case_id.nunique().eq(3).all()
    assert assign.loc[assign.status!='CLASSIFIED','cluster'].isna().all()
    trials=pd.read_csv(out/'stability.csv');assert len(trials)==720 and trials.valid_repeat.all()
    model='case__structure__kmedoids__k2'
    chosen=assign[assign.model_id==model].set_index('case_id')
    result=cf.loc[chosen.index].copy();result['cluster']=chosen.cluster
    repeated=result.groupby('cluster').n_runs__median.median().idxmax()
    labels={int(c):('反復多め・1回短め' if c==repeated else '反復少なめ・1回長め') for c in result.cluster.unique()}
    compact=pd.DataFrame(index=result.index)
    compact['candidate_group']=result.cluster.astype(int);compact['candidate_description']=result.cluster.map(labels)
    for field in ['active_s__median','n_runs__median','run_q50_s__median','active_duty__median',
                  'torque_q50__median','pedal2_q50__median','torque__eligible_unit_count','pedal2__eligible_unit_count']:
        compact[field]=result[field]
    compact.to_csv(out/'case_structure_candidates.csv',index_label='case_id')
    summaryrows=[]
    for cl,g in result.groupby('cluster'):
        summaryrows.append(dict(candidate_group=int(cl),description=labels[int(cl)],n_cases=len(g),
            typical_active_s=g.active_s__median.median(),typical_runs=g.n_runs__median.median(),
            typical_run_q50_s=g.run_q50_s__median.median(),typical_active_duty=g.active_duty__median.median()))
    pd.DataFrame(summaryrows).to_csv(out/'interpretable_structure_groups.csv',index=False)
    s=models[models.model_id==model].iloc[0]
    summary=dict(primary_scope=dict(source_case_audit=300,high_hp2_blocks=203,high_hp2_cases=67,
        numeric_supported_blocks=142,numeric_supported_cases=61,uncompared_numeric_blocks=61,uncompared_numeric_cases=6),
        conclusion='作動構造と数値水準はほぼ別軸。30秒高回転ブロックに条件付けた症例2群候補は解釈可能だが、普遍的・臨床的な型は未確立。',
        primary_display_model=model,structure_groups=summaryrows,
        structure_case_subsampling_ari=dict(median=s.stability_median,q25=s.stability_q25,q75=s.stability_q75),
        structural_numeric_distance_correlation=.033,nearest_case_top3_jaccard_structure_numeric=.043,
        same_hp_stage_structure_partition_ari=.0504,
        single_device=True,external_validation=False,original_sources_unchanged=True)
    (out/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    report='''高回転主体HP2の類似性評価 v4 — 作動構造・数値水準・前後文脈を分離

主な結論
高回転主体のHP2内には、作動の仕方が似た別症例が見つかる。
症例単位では「短い作動を多く繰り返す42例」と「1回の作動が比較的長く、回数が少ない25例」という探索的2群が得られた。
ただしこれは30秒以内のOFFをまとめた高回転ブロックの典型像であり、普遍的な処置分類ではない。
数値水準と作動構造の類似性はほぼ別軸。一つの距離に混ぜると群分けが不安定になる。
したがって、現時点では「作動の仕方 × トルク等の数値水準 × 前後の操作」の複数軸で表す方が適切。

対象と不支持例
元300症例の監査一覧を保持。以前確定した高回転主体203ブロック・67症例を固定して再評価した。
元のHP境界、v3のhigh_dominant等の判定、既存Two-stageの割当は変更しない。
主ブロックは同じHPで既知OFF30秒以下をまとめたもの。高回転は40000±10%、低<=5000、中は間の領域。
主numericは同じHP2run・秒binでRPMに対応するトルクとペダル2が各3bin以上ある142ブロック・61症例。
残る61ブロック/6症例は数値類似性では不支持であり、0や中央値で埋めて近隣やクラスを与えない。
作動構造は全203ブロック/67症例で比較。構造と数値の直接比較は共通142ブロック/61症例でも行った。
同一caseに1〜13ブロックがあり、独立203例ではない。67症例集約の解析を別に実施した。

3つの類似性
作動構造:実作動時間、経過時間中の作動割合、run数、1runの長さq50/q90、内部OFF長q50/q90。
数値水準:トルクとペダル2のq50/q90/IQR。バルク開量とRPMは原単位の監査/記述に残す。
前後文脈:直前のHP、既知間隔、症例中の位置、同症例にHP3後低回転などの特徴があるか。
RPMは既に高回転に限定され、数十rpmの差を小さなIQRで拡大するとノイズを過大評価するため主距離に入れない。
数値は観測値のみ。ペダルの値保持仕様は未確認なので、未観測を前方補完しない。

距離の作り方
時間・回数はlog1p、他は原値を使い、case均等重みの中央値/IQR等で尺度を揃えた。
正規化差を3scaleで上限1にし、同family内を平均、family同士は等重み。
トルクfamilyとペダル2familyも等重み。混合距離は構造0.5+数値0.5で、重み変更は感度解析として別保存。
症例集約は各blockを等重みの中央値にし、numericはそのsignalで3bin以上支持のあるblockだけを使った。
1blockしかない症例ではblock間IQRは不明。0と捏造せず、主case距離は中央値profileに限定しIQRは記述表へ保存した。
距離の小ささや1-distanceは同じ処置である確率ではない。

異なる症例の近隣を評価
各queryについて、異なる症例から近い上位3例を選択。同一症例内の別blockは近隣にしない。
近い実例・支持量を近づけたランダム対照・構造は近いが数値は遠い対照・その逆を実測図で比較。
対応は作動runと秒binに基づく。RPM/トルク/ペダル2/バルクの未観測は空白のまま、時間をDTWで伸縮しない。
実時間と累積HP2作動進捗の2図を作成し、同じ信号は共通の縦軸を使った。
作動構造と数値の距離相関は、共通142blockでSpearman0.033。top3近隣case集合Jaccard平均は0.043。
同じ作動パターンでもトルク・ペダルが異なる例、同じ数値水準でも作動長・繰り返しが異なる例を確認できる。

近隣の安定性の意味
case80%を再標本化し、scaleを毎回refit。参照と再fitの両方を同じ残存候補集合に制限して比較した。
これにより、単に近隣caseが再標本化で消えただけの変化を不安定とは数えない。
case集約の上位3保持率の中央値は、構造約0.969、数値約0.960。
ただしこれは同じ区間候補が残った条件でscaleを推定し直す安定性で、測定誤差や新規患者への再現性ではない。
数値の近隣が安定でも、離散的な群の境界が安定するとは限らない。

多数block症例の機会偏り
同じcaseのblockが多いほど、別caseのqueryに近いblockが1つ見つかる機会が増える。
caseの高block数と近隣選択頻度のSpearmanは構造0.909、数値0.575。
候補を各caseランダム1blockに揃える100反復では、通常のtop3とのcase均等平均Jaccardは構造0.258、数値0.356。
これはscale-refit安定性とは別の、候補機会数/同症例内の多様性に対する感度。
近隣によく選ばれるcaseをそれだけで「最も代表的」「最も再現性が高い」と呼ばない。
そのため主な解釈は67caseの集約解析を中心にする。case IDが異なることは独立患者であることの証明ではない。

候補分割の評価
block/case各単位で構造・数値・混合を分け、k-medoidsとaverage linkageのk2〜5を比較。
48候補から各view/手法の代表12モデルを保存。最小独立case数5、群サイズ、偏りを事前条件とした。
候補選択は8回case subsamplingとsilhouetteを用い、代表には別seed60回のcase80%再標本化を行った。
k-medoidsのscaleと目的関数はcase均等にし、多数blockcaseの影響を抑えた。averageは比較法で同じ重み付けではない。
720反復すべてで実kを維持。ARIの中央値だけでなく四分位・最小値を保存した。
各反復のkと特徴設計は固定であり、全パイプライン選択を毎回再実行した検証ではない。
silhouetteは特徴空間やcohortが違うmodel間の単純な性能ランキングにしない。

もっとも解釈しやすい構造の症例2群
case_structure_kmedoids2:
 反復多め・1回短め42例:典型block実作動49.1秒、5run、run長中央値5.86秒、作動割合80.6%。
 反復少なめ・1回長め25例:典型block実作動26.5秒、2run、run長中央値11.66秒、作動割合96.3%。
後者は1回の作動が長めであり、ブロック全体の総作動時間が長いという意味ではない。
内部ARI中央値0.926、IQR0.836〜1.000、silhouette0.287。
構造的な傾向差を説明する候補として有益だが、完全に分離した自然な2種類が存在する証明ではない。
実作動時間を除くと元分割とのARI0.826、OFF長を除くと0.573、run数を除くと0.669。
どの作動特徴を重視するかへの依存も残る。

block内の構造4群
k-medoids4群は26/61/66/50block、それぞれ21/31/38/30独立caseを含む。
長いblockで多数の短run、少数の長めrun、短くほぼ連続、短いrunの反復などの違いを示す。
内部ARI中央値0.770、IQR0.655〜0.827。1caseが複数の型のblockを持つため、4種類の症例ではない。

数値水準の群はより不確実
block numeric3群は84/50/8blockで、トルク中央値は概ね0.5/1.5/3。最小群は5case。
内部ARI中央値0.694で、確定3種類とはしない。
case numeric2群は26/35case、トルク中央値1.5/0.5などの違いがあるが、ARI中央値0.803に対してQ1は0.416。
トルクだけの分割と2信号分割のARIはblockで0.698、caseで0.537、ペダルだけでは約0.030/0.082。
主な数値軸はトルク寄りだが、複合的な処置型が確立したという意味ではない。
トルクの量子化された値の山や、記録方式も分布に影響し得る。

高silhouetteでも採用しない例
average numeric2群はblockで134対8、caseで56対5に偏り、事前の最大群割合90%条件に不合格。
混合averageも極小群に偏り、ARIが高い反復があっても推奨しない。
高silhouetteだけでクラスタの妥当性を主張しない。

混ぜた場合と観測支持の感度
等重み混合block2群のARI中央値は0.412で、構造単独より不安定。
支持条件を1/5/10binや観測fraction10/20%へ変えると対象数も大きく減り、数値分割の一致度も変化した。
例:各5binなら101block/50case、各10binなら58block/35case、fraction20%条件では27block/21case。
block numeric kmedoids3の支持変更ARIは約0.167〜0.942。全体に通用する頑健な数値3群は未確立。
混合の重みを構造25/75%にすると元のblock混合分割とのARIは0.306/0.096。
現時点で一つの万能距離に混ぜるより、構造・数値・文脈を別軸で併記する方が説明しやすい。
この支持閾値再fitはblock解析に実施。case numericを各閾値で再集約する検証までは行っていない。

大きい同HP段階との比較
主203高blockを包含する73stageを選び、元67caseを維持。stage自体が高回転主体のものだけへ再選定していない。
同じ中央値特徴定義・同じcase集合で比較すると、構造の近隣Jaccard0.061、分割ARI0.050。
数値の近隣Jaccard0.113、分割ARI0.213。
ただしstageには低中回転や長いOFFも加わり、問いそのものが広い操作段階へ変わる。
この差を単なる区切り誤差や、元の群が偽である証明とは解釈しない。
言えるのは、30秒高回転blockの典型像を、大きい混合HP段階へそのまま一般化できないということ。

観測状況と交絡
quality-only対照には記録bin数/fractionを使い、presenceが全1になるnumericcoreでも評価した。
case構造2群とquality分割のARI0.105、記録年NMI0.018、元症例長eta二乗0.002。
低い関連でも観測方式の影響が排除された証明ではない。block長とbin数は本質的にも関連する。
全例同一装置・既存選定cohortのため、他装置・施設・独立患者に対する外部妥当性は未検証。

使い方と次の判断
同じ高回転でも『短い作動を繰り返す/1回が長め』という構造軸と、『トルク等の数値水準』を分けて見られる。
まず原単位のcase図と異なる症例pairを見て、利用者が意味のある操作差か確認できる状態にした。
case_structure_candidates.csvは探索候補の所属で、診断・処置名・確定分類ではない。
近隣検索は似た観測例を探す道具として有用だが、近い例を選んだ図だけで未知症例への再現性を証明しない。
明瞭な離散群が無ければ、複数の連続的な操作軸として使うことも適切な結論。

保存と環境
元300scope、主203/67、numeric不支持、各modelのIDと理由を保持した。
既存venv使用、新規package/OS変更なし。CPU8worker、必要compactCSVのみ読込、wide case tablesは読まない。
元v3入力SHAを照合し、元の分類・判定・データが不変であることを確認した。
症例均等重みの経験分位点について、同一caseの全区間を同率複製しても距離が不変になる人工検証を実施。
浮動小数点のCDF境界丸めを修正し、初期出力と修正前sourceを保存して再計算した。
本データの8距離行列・12分割・720反復ARIは修正前後で同じだが、一部の症例重み付きprofile中央値を訂正した。
記述表の通常中央値と、重み付き経験CDFの分位点は偶数標本で定義上異なることがある。
process/に特徴修正、方法review、実装review、ケース対照、回復処理、最終判断を記録。
コードsnapshotと元単位対応・distance・raw支持・近隣差分を保存し、後から追えるようにした。
'''
    (out/'report.txt').write_text(report,encoding='utf-8')
    decisions='''類似性v4 判断履歴
1 高回転203block/67caseを固定。case67集約を主に解釈し、blockを独立203例にしない。
2 構造/数値/文脈を別距離、RPMはaudit。主numericはtorque/pedal2各3RPM対応bin。
3 feature担当がcase numericに1binblockを混ぜないよう修正し、before版もprocessに保持。
4 不支持はNaN/NO_NUMERIC_SUPPORT。主numeric142block61case、同stage包含73stage67case。
5 raw boolean文脈の差分CSV出力でTypeError。models/距離/近隣は計算済みで不変。
  raw値をfloatへcastする記述exportだけをresume-finalizeし、不要な再fitは行わなかった。
6 independent reviewで構造群のnumericプロフィールの少数N混入を指摘。
  cluster_profiles_supportedで各code3bin未満をmask、n_supported_casesを別に表示するよう修正。
7 case構造42/25は説明可能だが、段階や特徴選択への依存があり確定taxonomyとはしない。
8 numeric/combinedは不安定。高silhouetteでも極小群・偏り基準不合格は採用しない。
9 block数の多いcaseは近隣候補の機会が多い。case集約と1case1候補抽出感度を追加。
10 stageは同67caseを維持するが低中回転も含む。別の問いへの表現変更として解釈。
11 近いpair・支持matched random・2軸が食い違うpairを原単位で並べ、閾値/重みを良い結果へ後調整しない。
12 最終表示はcase構造2群候補と独立した負荷軸。内部安定性・近隣の条件付き保持・外部妥当性を混同しない。
13 症例均等weighted empirical CDFの丸め境界を人工複製テストで発見、精度tolを修正。
  全初期derived CSV/距離/sourceをprocess/pre_precision_fix、図をzipへ保全。原特徴データは不変。
  再計算で実データD8行列・全12分割・720ARIは同じ、case-balancedプロフィール中央値の一部は訂正。
  Fraction厳密CDFによる独立1400チェックと複製不変性を通過し、修正後を最終報告とする。
'''
    (out/'process/decisions.txt').write_text(decisions,encoding='utf-8')
    code=Path(__file__).resolve().parent
    validation=dict(primary_units=203,primary_cases=67,numeric_units=142,numeric_cases=61,
        original_sources_unchanged=True,source_tables_verified=len(meta['source_sha256']),model_candidates=48,models=12,
        case_subsampling_trials=720,all_trials_valid=True,neighbors_always_distinct_cases=True,
        duplicate_assignment_keys=0,duplicate_neighbor_keys=0,uncompared_rows_unassigned=True,
        python_executable=sys.executable,scripts_sha256={p.name:sha(p) for p in code.glob('*.py')})
    (out/'quality_checks.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
