"""Scientific synthesis, comparable assignments, and reproducibility checks."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from cluster_observed_v2 import prepare


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();v2=out/'recording_semantics_v2'
    root=out.parent.parent
    manifest=pd.read_csv(out/'case_manifest.csv')
    oldassign=pd.read_csv(out/'assignments_long.csv')
    newassign=pd.read_csv(v2/'assignments_long.csv')
    models=pd.read_csv(out/'selected_models.csv')
    newmodels=pd.read_csv(v2/'selected_models.csv')
    f=pd.read_csv(v2/'features.csv').set_index('case_id')
    cat=pd.read_csv(v2/'feature_catalog.csv',dtype={'code':str})
    audit=pd.read_csv(out/'signal_audit.csv',dtype={'code':str})
    raw,analysiscat,ok=prepare(f,cat,audit)
    source_checks=[]
    for row in manifest.itertuples():
        source_checks.append(dict(case_id=row.case_id,source_sha256_matches=sha(row.source_path)==row.source_sha256))
    assert all(r['source_sha256_matches'] for r in source_checks)
    pd.DataFrame(source_checks).to_csv(out/'source_integrity.csv',index=False)
    # Check the existing Two-stage source tree against the prior read-only replay snapshot.
    old_snapshot=root/'Output/chair_two_stage_reproducibility_20261104/input_sha256_before.json'
    original_results_unchanged=True
    for rel,expected in json.loads(old_snapshot.read_text()).items():
        assert sha(root/rel)==expected,rel
    # Every model preserves all 300 scope rows; absence of data is never a class.
    for frame in [oldassign,newassign]:
        assert not frame.duplicated(['case_id','model_id']).any()
        for mid,g in frame.groupby('model_id'):
            assert len(g)==300 and set(g.case_id)==set(manifest.case_id)
            assert g.loc[g.status!='CLASSIFIED','cluster'].isna().all()
    assert (len(newassign.query("model_id=='observed__kmedoids__k3' and status=='CLASSIFIED'")))==287
    reps=pd.read_csv(v2/'representatives.csv')
    for (mid,cl),g in reps.groupby(['model_id','cluster']):
        assert g.is_medoid.sum()==1 and bool(g.sort_values('rank').iloc[0].is_medoid)
        valid=newassign[(newassign.model_id==mid)&(newassign.cluster==cl)]
        assert set(g.case_id).issubset(set(valid.case_id))
    # Consolidated 17-model comparison, using exactly the intersection of classifiable cases.
    allassign=pd.concat([oldassign,newassign],ignore_index=True)
    wide=allassign.pivot(index='case_id',columns='model_id',values='cluster').reindex(manifest.case_id)
    wide.to_csv(out/'assignments_comparison_wide.csv',index_label='case_id')
    common=wide.dropna()
    agreement=pd.DataFrame([[adjusted_rand_score(common[a],common[b]) for b in common]
                             for a in common],index=common.columns,columns=common.columns)
    agreement.to_csv(out/'all_model_comparison_ari_common.csv',index_label='model_id')
    common.index.to_series().to_csv(out/'all_model_common_case_ids.csv',index=False,header=['case_id'])
    mainassign=newassign[newassign.model_id=='observed__kmedoids__k3'].copy()
    final=manifest.merge(mainassign[['case_id','cluster','status']],on='case_id',validate='one_to_one')
    scope=pd.read_csv(v2/'cohort_scope.csv')
    extra=[col for col in scope if col not in final and col!='v2_status']
    if extra:
        final=final.merge(scope[['case_id']+extra],on='case_id',validate='one_to_one')
    final['interpretation']='EXPLORATORY_OPERATION_PATTERN; NOT_CLINICAL_TREATMENT_LABEL'
    final.to_csv(out/'case_assignments_v2_candidate.csv',index=False)
    # Raw interpretation statistics use the same support masking as model fitting.
    class_rows=[]
    for model in newmodels.itertuples():
        group=newassign[(newassign.model_id==model.model_id)&newassign.cluster.notna()].set_index('case_id')
        for cl,part in group.groupby('cluster'):
            for feature in raw:
                values=raw.loc[part.index,feature].dropna()
                class_rows.append(dict(model_id=model.model_id,cluster=int(cl),feature=feature,
                    n_cases=len(part),n_observed=len(values),raw_median=values.median() if len(values) else np.nan,
                    raw_q25=values.quantile(.25) if len(values) else np.nan,
                    raw_q75=values.quantile(.75) if len(values) else np.nan))
    profiles=pd.DataFrame(class_rows)
    profiles.to_csv(v2/'class_profiles.csv',index=False)
    parent=mainassign[mainassign.cluster.notna()].set_index('case_id')
    names={1:'HP作動割合低め・回転数が混在する候補',2:'回転数高値・HP状態2寄りの候補',
           3:'回転数低値・HP状態3寄りの候補'}
    selected_keys={'回転数中央値':'E20742__q50','回転数上位値':'E20742__q90',
        'トルク中央値':'E20743__q50','バルク開量中央値':'E20108__q50',
        '作動中HP状態2割合':'E20039__state2_fraction','作動中HP状態3割合':'E20039__state3_fraction',
        'シリンジ取上げ割合':'E20101__pickup_fraction','スケーラー作動割合':'E20774__active_fraction',
        '作動中スケーラー出力設定':'E20776__active_setting_timeweighted_q50'}
    interpreted=[]
    for cl,group in parent.groupby('cluster'):
        row=dict(cluster=int(cl),n_cases=len(group),candidate_description=names[int(cl)])
        for label,key in selected_keys.items():
            values=raw.loc[group.index,key].dropna()
            row[key]=values.median() if len(values) else np.nan
            row[key+'__n']=len(values)
        hp_known=f.loc[group.index,'E20039__known_fraction']>=.2
        hpactive=f.loc[group.index,'E20039__active_fraction'].loc[hp_known]
        row['HP_nonzero_fraction']=hpactive.median()
        interpreted.append(row)
    descriptions=pd.DataFrame(interpreted)
    descriptions.to_csv(v2/'interpretable_class_summary.csv',index=False)
    comparisons=pd.concat([models.assign(version='v1_observed_bins'),newmodels.assign(version='v2_recording_semantics')],ignore_index=True)
    comparisons.to_csv(out/'all_model_summary.csv',index=False)
    candidates=newmodels.set_index('method').loc['kmedoids']
    counts=mainassign.groupby('status').size().to_dict()
    summary=dict(main_report='recording_semantics_v2/report.html',
        conclusion='解釈可能な3群の操作傾向候補を得たが、観測支持条件への感度が残り、頑健な処置分類は未確立',
        preferred_display_model='observed__kmedoids__k3',scientific_status='EXPLORATORY_CANDIDATE',
        requested_cases=300,v1_classified=296,v1_numeric_only=289,v2_status_counts=counts,
        common_cases_all_17_models=len(common),class_counts={str(int(cl)):int(n) for cl,n in
            mainassign[mainassign.cluster.notna()].cluster.value_counts().sort_index().items()},
        v2_median_ari=float(candidates.stability_median),v2_ari_q25=float(candidates.stability_q25),
        v2_ari_q75=float(candidates.stability_q75),v2_silhouette=float(candidates.silhouette),
        v2_quality_ari=float(candidates.quality_ari),
        scaler_definition='変更時記録、次の記録まで有効。作動中の設定を時間加重集計。',
        syringe_definition='0取り上げ、1格納。取り上げ時間は噴射時間ではない。',
        original_sources_unchanged=True,external_validation=False)
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (v2/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

    text='''DTWを使わない300症例の操作パターン解析 — 最終評価

結論
利用者のイベント定義を反映した第2版で、解釈できる3群の操作傾向候補を得た。
ただし、観測支持条件を変えると所属が変わるため、頑健な分類や処置の識別が確立したとは言えない。
新しい信号定義を反映した recording_semantics_v2/report.html を主報告とする。
第1版の15モデル・図・割当は参考比較として保存し、上書き・隠蔽していない。

解析対象と境界
既存Hybrid300のselected_cases.csvを固定。最初の90012=1から最後の90012=0までを継続した。
途中消灯・再点灯を含める利用者の指定であり、点灯中だけに区間を再定義していない。
300例は同一装置C512KC1503、症例長約1549〜2598秒に限られる選定集団で、全ログの無作為標本ではない。
10コード全未記録の4例は、利用者の「スキップして」に基づきNO_TARGET_SIGNALとして保持した。
第1版は操作/組合せ296例、数値水準のみ289例。数値情報が全欠測の追加7例を補完だけで分類しないよう修正した。
第2版は、HP選択・作動状態の既知時間が各20%以上、少なくとも1種類の支持された数値信号がある287例。
残り9例はINSUFFICIENT_COMPARABLE_SUPPORTであり、不使用や異常のクラスではない。
全300例の所属・未分類理由はcase_assignments_v2_candidate.csvと各版のcohort_scopeに保存した。

記録意味に関する重要な更新
利用者確認: 20776は設定値が変化した時に記録され、次の記録まで値が有効。
したがって、20776を多数の連続測定値と同様に扱い、記録点数だけで信頼性を判断するのは適切でない。
第2版では20774=1かつ20776設定が既知の区間を実時刻で切り、設定水準を時間加重した。
設定を計算できた225例すべてで、既知作動時間の設定値は100%観測によりカバーされていた。
利用者確認: 20101=0は取り上げ、1は格納。取り上げ割合・取上げ/返却回数を導出した。
取り上げ時間は実際の噴射時間・治療時間と同一ではない。
他の数値コードは値保持の仕様が未確認なので、観測された1秒binの中央値分布を維持した。
HP状態値1/2/3の追加の機器意味や処置名は推測していない。

第1版の比較と改善過程
153特徴を抽出し、5表現×3手法×k2〜8の105候補から、各表現/手法の代表15モデルを比較した。
表現: 数値水準、操作構成、両者の組合せ、作動HP条件付き数値の組合せ、前中後の粗い構成。
手法: KMeans、Ward階層法、対角共分散Gaussian mixture。
Gaussian mixtureは各軸の分散を推定するため、前処理でのブロック重みはKMeans/Wardと同じ寄与を保証しない。
選択は事前保存したサイズ・偏り条件とsilhouette/探索ARIスコアによる。後から高得点になるよう重みは変更していない。
15代表モデルは各60回のケース80%サブサンプリングを行い、補完・標準化を毎回refitした。
全900反復が収束・実クラス数条件を満たした。中央値だけでなく分位点と最小値も保存した。
第1版KMeansでは数値水準2群、操作構成6群、組合せ2群などの内部安定性は高く見えた。
しかし、低観測特徴の除外条件を変える感度解析では、操作6群の一致ARIは0.368へ低下した。
組合せKMeans2群は厳しい支持条件でARI0.003まで変化した。高い内部ARIだけでは頑健性を保証しない。
第1版のスケーラー出力の記録点分布は、設定持続時間を反映しないため新版と区別する。
全欠測数値を持つ症例が組合せ空間のMedoidに選ばれる問題も代表図で露呈した。
これらを隠さず残し、意味補正と補完を避ける第2版へ進めた。

第2版の方法
168特徴の元表からlevels/operationsを使い、重複する補数や旧解釈列はauditへ分離した。
連続数値は少なくとも10観測bin、状態は既知時間20%以上を支持条件とする。
この数値閾値は解析上の感度条件であり、機器に対して検証済みの品質基準ではない。
20776設定は変更時記録なので、この観測bin数閾値を適用しない。
元の欠測値はNaNで保存し、距離計算でも中央値や0へ補完しない。
両症例で観測できる特徴の差をコード単位に集計し、数値と操作の2系統を均等に組み合わせた。
各比較には少なくとも2コードが共通し、共通コード数中央値は7。
ただし903/41041ペア(2.20%)では共通数値信号がなく、操作状態だけの距離になった。
ペアごとに比較するコードが異なるため、同じ物理量を一律に比べるEuclidean距離ではない。
Wardは使わず、k-medoidsとaverage linkageを比較。k2〜8の14候補を事前規則で評価した。
各代表モデルの60回サブサンプリングではスケールも再推定した。実観測症例をMedoidとして保存した。

第2版の結果
k-medoids3群は73/64/150例、silhouette=0.158。
ケースサブサンプリングARI中央値=0.741、四分位範囲0.509〜0.943、最小0.223。
一定の内部再現性はあるが、分離は弱く、反復による変動も大きい。
average linkage2群は281/6例と偏り、事前サイズ・均衡基準を満たさず不採用とする。
silhouetteが0.305と高くても、これを良い分類の根拠にしない。

3群の説明（すべて観測支持を満たす症例での原単位集計）
C1: 73例。回転数中央値の群内中央値16961（観測48例）、出力設定7（観測43例）。
    HP非0状態割合は既知時間中の中央値15.2%。回転数が低値から高値まで混在する、作動割合低めの候補。
C2: 64例。回転数34948（55例）、出力設定9（35例）、HP非0割合24.5%。
    HP状態2の割合が相対的に高い候補。スケーラー情報は全64例を代表しない。
C3: 150例。回転数999（145例）、出力設定5（147例）、HP非0割合44.0%。
    HP状態3割合が高く、スケーラー作動割合も高めの候補。
各値は全観測点を一括平均したものではなく、各症例の代表値の群内中央値。
シリンジ取上げ割合はC1=.140(n66)、C2=.229(n62)、C3=.068(n61)。
C3のシリンジ状態既知時間中央値は5.8%しかなく、シリンジ情報が乏しい多数の症例にはこの値を一般化しない。
全特徴の原単位中央値・四分位点・観測症例数はinterpretable_class_summary/class_profilesを参照。
第2版C1のMedoidは回転数未観測であり、回転数の代表波形とは言えない。
補完を避けても、ペアごとに共通信号が異なる距離では全信号を代表するMedoidが保証されないため、複数症例を併記した。

残る問題と採用判断
第2版3群と、連続数値支持条件を5binへ変更した結果の一致ARI=.706、20binでは.401。
状態既知時間50%以上に限定した条件では270共通症例でARI=.751。
したがって、この3群も「頑健な最終クラス」とは呼べず、探索候補として保持する。
支持条件を変えると使える特徴、分布、再標準化、症例も変わるため、変動の原因を欠測補完だけに帰属できない。
観測状況だけの対照分類とのARIは.363。低い値ではなく、観測様式の影響は残り得る。
記録年NMI=.018、症例長eta二乗=.005で、この選定集団内の年/長さとの単純な関連は小さい。
全例が同一装置であり、装置差・施設差・独立患者に対する再現性は検証できない。
現時点の採用は、3群を可視化・症例確認のための操作傾向候補として使うことに限る。
処置名への対応、診断や臨床的有用性は教師情報がなく評価していない。
次の優先事項は、代表症例を利用者が確認し、HP状態値の意味と他数値の記録仕様を照合すること。
データ数を増やすだけで、これらの意味・欠測の問題が解消するとは限らない。

比較の読み方
同じ症例で15モデルと新版2モデルの所属を横並びにし、共通287症例で全17モデルのARIを保存した。
特徴空間が違うモデルのsilhouetteは、直接同一尺度の性能ランキングにしない。
旧Two-stageはActivity単位、本解析は照明区間の症例単位なので、精度の単純な優劣比較はしていない。
PCA/t-SNE/MDSの分離は参考可視化であり、クラスの正しさの証明ではない。
内部安定性は固定した特徴・k・支持条件に条件付けた同一集団内の検証で、外部検証とは区別する。

環境と保存
既存venv ~/venvs/chair_c31、CPU8workerを使用。新規パッケージ導入やOS設定変更なし。
GPUはドライバ通信不可だったため使用せず、環境の変更は行わなかった。
元イベントは必要列だけchunk読込。wide case tableを全件メモリ展開していない。
元300症例のイベントファイルSHA、既存Two-stage結果、旧replayスクリプトの不変性を確認した。
利用者への質問・回答、独立レビュー、修正と追加解析の理由はprocess/に個別記録した。

参考にした実装一次資料
scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html
scikit-learn 1.7 SimpleImputer: https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html
Raw values and missingness are retained in exported audits; algorithm-only v1 imputation is not an OFF observation.
'''
    (out/'report.txt').write_text(text,encoding='utf-8')
    (v2/'report.txt').write_text(text,encoding='utf-8')
    (out/'process/final_decisions.txt').write_text(
        '最終判断: 意味を訂正した第2版を主表示とし、kmedoids3は探索候補、average2は不採用。\n'
        '第1版の高ARIを確定分類と呼ばない。観測支持感度と代表症例の情報不足が判明したため。\n'
        '利用者確認20776保持設定/20101=0取上げ1格納を新versionへ反映。旧版は比較保存。\n'
        'v2の追加9支持不足例はcase_scope全300に明示し、結果から黙って落とさない。\n'
        '両方式のsilhouetteを直接競争させず、異なるcohort/特徴/距離の変更を明記する。\n'
        'スケール・支持閾値は感度を見せる目的で並列保存し、最良スコアとなる閾値を採択しない。\n',encoding='utf-8')
    code=Path(__file__).resolve().parent
    validation=dict(requested_cases=300,v1_models=15,v2_models=2,all_model_common_cases=len(common),
        source_event_sha256_checked=300,source_event_hashes_unchanged=True,
        existing_twostage_and_replay_unchanged=True,duplicate_assignment_keys=0,
        unsupported_cases_have_no_cluster=True,v2_status_counts=counts,
        python_executable=sys.executable,platform=platform.platform(),
        scripts_sha256={p.name:sha(p) for p in code.glob('*.py')})
    (out/'final_validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
