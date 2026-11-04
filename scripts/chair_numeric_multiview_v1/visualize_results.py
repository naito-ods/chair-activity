"""Auditable scientific figures and local interactive report for non-DTW models.

Reads compact feature/audit tables only. Does not fit or modify any partition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

NAMES = {'20011': '選択HP', '20039': '作動中HP', '20742': '回転数',
         '20108': 'バルク開量', '20743': 'トルク', '20101': 'シリンジ状態（0取上げ/1格納）',
         '20774': 'スケーラー作動', '20776': 'スケーラー出力設定',
         '10014': 'ペダル1', '10015': 'ペダル2'}
NUMERIC = ['20742', '20108', '20743', '20776', '10014', '10015']
STATES = ['20011', '20039', '20101', '20774']
VIEWS = {'levels': '数値水準', 'operations': '操作構成', 'combined': '数値＋操作',
         'conditional': '作動状態との組合せ', 'coarse': '大きな段階の構成',
         'quality': '観測状況のみ（負対照）'}


def safe(value):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(value))


def asnum(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=140)
    parser.add_argument('--html-only', action='store_true', help='Refresh report after the evaluation text is ready')
    args = parser.parse_args()
    out = args.output.resolve()
    figs = out / 'figures'
    figs.mkdir(exist_ok=True)
    process = out / 'process'
    process.mkdir(exist_ok=True)

    def log(event, **details):
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), event=event, **details)
        with (process / 'visualization_log.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def load(name, required=True):
        path = out / name
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return pd.DataFrame()
        return pd.read_csv(path)

    log('start', dpi=args.dpi, policy='Compact audit tables only; no original wide tables')
    manifest = load('case_manifest.csv')
    audit = load('signal_audit.csv')
    features = load('features.csv').set_index('case_id')
    catalog = load('feature_catalog.csv')
    selected = load('selected_models.csv')
    candidates = load('model_candidates.csv')
    assignments = load('assignments_long.csv')
    embedding = load('embedding.csv')
    representatives = load('representatives.csv')
    coarse = load('coarse_profiles.csv')
    stability = load('stability_repeats.csv')
    audit['code'] = audit.code.astype(str).str.replace(r'\.0$', '', regex=True)
    coarse['code'] = coarse.code.astype(str).str.replace(r'\.0$', '', regex=True)
    catalog['code'] = catalog.code.astype(str).str.replace(r'\.0$', '', regex=True)
    assert not manifest.case_id.duplicated().any()
    assert not assignments.duplicated(['case_id', 'model_id']).any()
    if 'date' not in manifest:
        manifest['date'] = manifest.case_id.str.split('__').str[0]
    manifest['year'] = manifest.date.astype(str).str[:4]
    manifest_index = manifest.set_index('case_id')
    figure_index = []
    if args.html_only:
        figure_index = pd.read_csv(out / 'figure_index.csv').fillna('').to_dict('records')
        supplemental_figures(out, figure_index, args.dpi, log)
        pd.DataFrame(figure_index).to_csv(out / 'figure_index.csv', index=False)
        build_html(out, manifest, audit, selected, assignments, embedding, figure_index, log)
        log('html_refresh_complete', n_figures=len(figure_index))
        return
    with tempfile.TemporaryDirectory(prefix='chair-figures-') as tmp:
        os.environ['MPLCONFIGDIR'] = tmp
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D
        plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                             'axes.unicode_minus': False, 'figure.facecolor': '#ffffff',
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'savefig.facecolor': '#ffffff'})
        palette = plt.get_cmap('tab10')

        def save(fig, name, title, category='overview', model_id='', caption=''):
            path = figs / name
            fig.savefig(path, dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            figure_index.append(dict(path=str(path.relative_to(out)), title=title,
                category=category, model_id=model_id, caption=caption))

        codes = list(NAMES)
        matrix = audit.pivot(index='case_id', columns='code', values='observed_fraction')
        ids = manifest.sort_values('date').case_id.tolist()
        matrix = matrix.reindex(index=ids, columns=codes)
        present = audit.assign(present=audit.n_events.gt(0)).groupby('code').present.sum().reindex(codes)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.1, 2]})
        labels = [f'{c} {NAMES[c]}' for c in codes]
        bars = axes[0].barh(labels[::-1], present.to_numpy()[::-1], color='#427ca8')
        for bar, n in zip(bars, present.to_numpy()[::-1]):
            axes[0].text(n+2, bar.get_y()+bar.get_height()/2, str(int(n)), va='center')
        axes[0].set_xlim(0, len(manifest)+25)
        axes[0].set_xlabel('イベントが記録された症例数 / ' + str(len(manifest)))
        axes[0].set_title('コード別の記録症例数')
        image = axes[1].imshow(matrix.to_numpy(float), cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        axes[1].set_xticks(range(len(codes)), labels, rotation=60, ha='right')
        axes[1].set_ylabel('症例（記録日の昇順）')
        axes[1].set_title('数値の観測時間比率：未記録とOFFを区別')
        fig.colorbar(image, ax=axes[1], label='イベントを観測した秒bin / 症例の秒bin')
        fig.suptitle('300症例の入力監査：10イベントの記録状況', fontsize=16)
        fig.tight_layout()
        save(fig, '01_signal_coverage.png', '10イベントの記録状況', caption='未記録は作動していなかったことを意味しません。状態保持による既知時間と、実際にイベントを観測した時間は別です。')

        cats = catalog[catalog.view.isin(['levels', 'operations', 'conditional'])]
        cols = [c for c in cats.feature if c in features]
        missing = features.reindex(ids)[cols].isna().to_numpy()
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(missing, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        positions = np.linspace(0, max(0, len(cols)-1), min(20, len(cols))).astype(int)
        ax.set_xticks(positions, [cols[i] for i in positions], rotation=60, ha='right', fontsize=8)
        ax.set_ylabel('症例（記録日の昇順）')
        ax.set_title('特徴量の欠測：黒＝未観測、白＝値あり', fontsize=15)
        fig.tight_layout()
        save(fig, '02_feature_missingness.png', '特徴量の欠測', caption='raw特徴の欠測を表示しています。モデル内部の補完値ではありません。')

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(manifest.duration_s/60, bins=20, color='#427ca8', edgecolor='white')
        axes[0].set_xlabel('照明の最初ON→最後OFF（分）')
        axes[0].set_ylabel('症例数')
        yy = manifest.groupby('year').size()
        axes[1].bar(yy.index, yy.values, color='#55a48b')
        axes[1].set_ylabel('症例数')
        nn = audit.assign(present=audit.n_events.gt(0)).groupby('case_id').present.sum()
        axes[2].hist(nn, bins=np.arange(-.5, 11.5, 1), color='#d39947', edgecolor='white')
        axes[2].set_xticks(range(11))
        axes[2].set_xlabel('記録された対象コードの種類数')
        axes[2].set_ylabel('症例数')
        fig.suptitle('対象集団の構成：時間・記録年・情報量', fontsize=16)
        fig.tight_layout()
        save(fig, '03_cohort.png', '対象集団の構成', caption='全症例が同一装置のログである場合、装置をまたぐ再現性は評価できません。')

        viewlist = list(dict.fromkeys(candidates.view.astype(str)))
        fig, axes = plt.subplots(len(viewlist), 2, figsize=(12, max(4, 2.8*len(viewlist))), squeeze=False)
        for row, view in enumerate(viewlist):
            for method, g in candidates[candidates.view == view].groupby('method'):
                g = g.sort_values('k')
                axes[row, 0].plot(g.k, g.silhouette, marker='o', label=method)
                stabcol = 'stability_median' if 'stability_median' in g else None
                if stabcol:
                    axes[row, 1].plot(g.k, g[stabcol], marker='o', label=method)
            for col in range(2):
                axes[row, col].set_title(VIEWS.get(view, view) + ('：分離度' if col == 0 else '：内部安定性'))
                axes[row, col].set_xlabel('クラス数 k')
                axes[row, col].grid(alpha=.2)
                axes[row, col].legend(fontsize=8)
            axes[row, 0].set_ylabel('Silhouette')
            axes[row, 1].set_ylabel('症例サブサンプリング ARI 中央値')
            axes[row, 1].set_ylim(-.05, 1.05)
        fig.tight_layout()
        save(fig, '04_candidate_comparison.png', '特徴セット・手法・クラス数の比較', caption='分離度と内部安定性は異なる指標です。高い分離度だけで処置クラスと判断しません。')

        order = selected.model_id.astype(str).tolist()
        fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(order)*.35)), sharey=True)
        for i, row in selected.reset_index(drop=True).iterrows():
            vals = stability[stability.model_id == row.model_id]
            scol = 'ari' if 'ari' in vals else 'ari_to_reference'
            if scol in vals and len(vals):
                axes[0].boxplot([vals[scol].dropna()], positions=[i], vert=False, widths=.6,
                                patch_artist=True, boxprops={'facecolor': '#80b2d2'}, showfliers=True)
            axes[1].barh(i, asnum(row.get('silhouette')), color='#55a48b')
            axes[2].barh(i, asnum(row.get('quality_ari')), color='#d39947')
        axes[0].set_yticks(range(len(order)), order, fontsize=8)
        axes[0].set_xlim(-.1, 1.05)
        axes[0].set_xlabel('内部安定性 ARI（分布）')
        axes[1].set_xlabel('Silhouette')
        axes[2].set_xlabel('観測状況だけの分類との ARI')
        axes[2].set_xlim(-.1, 1.05)
        for ax in axes:
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('代表モデルの比較：安定性・分離度・欠測との関連', fontsize=16)
        fig.tight_layout()
        save(fig, '05_selected_model_metrics.png', '代表モデルの評価', caption='観測状況の負対照との一致が高い場合、操作そのものより記録状況で分かれている可能性があります。ARIは外部検証ではありません。')

        agreementpath = out / 'model_comparison_ari_common_cohort.csv'
        common_agreement = agreementpath.exists()
        if not common_agreement:
            agreementpath = out / 'model_comparison_ari.csv'
        if agreementpath.exists():
            agreement = pd.read_csv(agreementpath, index_col=0).reindex(index=order, columns=order)
            fig, ax = plt.subplots(figsize=(max(9, len(order)*.6), max(8, len(order)*.55)))
            im = ax.imshow(agreement.to_numpy(float), cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(order)), order, rotation=65, ha='right', fontsize=8)
            ax.set_yticks(range(len(order)), order, fontsize=8)
            for i in range(len(order)):
                for j in range(len(order)):
                    val = agreement.iloc[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if val > .6 else '#203040')
            fig.colorbar(im, ax=ax, label='モデル間 ARI')
            ax.set_title('同じ症例が同じようにまとまるか：モデル間の分類一致', fontsize=15)
            fig.tight_layout()
            save(fig, '06_model_agreement.png', 'モデル間の分類一致', caption='クラス番号を直接一致させず、症例のまとまり方をARIで比較しています。'+('全モデル共通の症例集合に揃えた比較です。' if common_agreement else '比較可能な症例の共通部分で比較しています。'))

        # Feature location and scale used only for a readable profile graphic.
        # Raw class medians stay unchanged and remain available in the audit CSV.
        feature_medians = features.median()
        feature_iqrs = features.quantile(.75) - features.quantile(.25)
        feature_iqrs = feature_iqrs.where(feature_iqrs.abs() > 1e-10, features.std()).replace(0, np.nan)
        profilecols = [f for f in catalog[catalog.view.isin(['levels', 'operations'])].feature if f in features]
        profile_limits = {}
        for code in NUMERIC:
            vals = pd.to_numeric(coarse.loc[coarse.code==code,'raw_median'], errors='coerce').dropna()
            if len(vals):
                lower = min(0., float(vals.min()))
                upper = float(vals.max())
                span = max(upper-lower, 1.)
                profile_limits[code] = (lower-.02*span if lower<0 else 0., upper+.07*span)
        log('overview_figures_complete', n_figures=len(figure_index), n_models=len(order))

        for modelrow in selected.itertuples():
            model_id = str(modelrow.model_id)
            slug = safe(model_id)
            ass = assignments[(assignments.model_id == model_id) & assignments.cluster.notna()].copy()
            ass['cluster'] = ass.cluster.astype(int)
            clusters = sorted(ass.cluster.unique())
            joined = ass[['case_id', 'cluster']].merge(manifest, on='case_id', validate='one_to_one')
            emb = embedding[embedding.model_id == model_id].merge(ass[['case_id', 'cluster']], on='case_id', validate='one_to_one')
            embtypes = [('pc1', 'pc2', 'PCA：特徴量空間の線形投影'), ('tsne1', 'tsne2', 't-SNE：近傍構造の参考表示')]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, (xcol, ycol, title) in zip(axes, embtypes):
                if xcol not in emb or ycol not in emb:
                    ax.text(.5, .5, '投影データなし', ha='center', transform=ax.transAxes)
                    continue
                for cl in clusters:
                    g = emb[emb.cluster == cl]
                    ax.scatter(g[xcol], g[ycol], s=22, alpha=.75, color=palette((cl-1) % 10), label=f'C{cl} (n={len(g)})', linewidths=0)
                ax.set_title(title)
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(alpha=.15)
            axes[1].legend(loc='best', fontsize=8)
            fig.suptitle(model_id, fontsize=15)
            fig.tight_layout()
            save(fig, f'{slug}_embedding.png', '症例の分布', 'model', model_id,
                 '2次元投影は表示のためで、投影上の見た目だけで分離や妥当性を判断しません。モデルは元の特徴量で分類しています。')

            gfeatures = features.reindex(ass.case_id).copy()
            gfeatures['cluster'] = ass.cluster.to_numpy()
            values = gfeatures.groupby('cluster')[profilecols].median().reindex(clusters)
            scaled = values.subtract(feature_medians, axis=1).divide(feature_iqrs, axis=1).clip(-3, 3)
            # Keep informative columns and group their labels by blocks.
            informative = scaled.var(axis=0, skipna=True).sort_values(ascending=False)
            keep = informative[informative.notna()].index[:min(40, len(informative))].tolist()
            keep = [f for f in profilecols if f in keep]
            fig, ax = plt.subplots(figsize=(max(12, len(keep)*.34), max(3, len(clusters)*.65)))
            data = np.ma.masked_invalid(scaled[keep].to_numpy(float))
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('#c4cbd2')
            im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-3, vmax=3)
            counts = ass.groupby('cluster').size()
            ax.set_yticks(range(len(clusters)), [f'C{c} (n={counts[c]})' for c in clusters])
            ax.set_xticks(range(len(keep)), keep, rotation=65, ha='right', fontsize=7)
            ax.set_title(model_id + '：クラスの特徴（raw中央値の相対比較）', fontsize=14)
            fig.colorbar(im, ax=ax, label='(クラスraw中央値 − 全体中央値) / 全体IQR；灰色＝未観測')
            fig.tight_layout()
            save(fig, f'{slug}_feature_profiles.png', 'クラス別の特徴', 'model', model_id,
                 '図示のみ全体IQRで尺度を揃えています。原単位の値は次の分布図とclass_profiles.csvで確認できます。')

            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, code in zip(axes.ravel(), NUMERIC):
                f = f'E{code}__q50'
                if f in features:
                    for pos, cl in enumerate(clusters):
                        cids = ass.loc[ass.cluster == cl, 'case_id']
                        vals = features.reindex(cids)[f].dropna().to_numpy()
                        if len(vals):
                            ax.boxplot([vals], positions=[pos], widths=.5, showfliers=False,
                                       medianprops={'color':'black'}, patch_artist=True,
                                       boxprops={'facecolor': palette((cl-1)%10), 'alpha':.3})
                            ax.scatter(pos + np.linspace(-.16, .16, len(vals)), vals, s=10,
                                       alpha=.5, color=palette((cl-1)%10), linewidths=0)
                        ax.text(pos, .98, f'{len(vals)}/{int(counts[cl])}', transform=ax.get_xaxis_transform(),
                                ha='center', va='top', fontsize=8)
                ax.set_xticks(range(len(clusters)), [f'C{c}' for c in clusters])
                ax.set_title(f'{code} {NAMES[code]}')
                ax.set_ylabel('症例内観測秒binの中央値（記録値）')
                ax.grid(axis='y', alpha=.2)
            fig.suptitle(model_id + '：数値水準の比較（点＝1症例、上段数字＝観測例/全例）', fontsize=15)
            fig.tight_layout()
            save(fig, f'{slug}_raw_numeric_levels.png', '原単位の数値水準', 'model', model_id,
                 '1秒binの観測値から求めた症例中央値です。未観測例を0として箱ひげに含めていません。')

            fig, axes = plt.subplots(2, 2, figsize=(13, 7))
            for ax, code in zip(axes.ravel(), STATES):
                statecols = [f'E{code}__state{s}_fraction' for s in ([0, 1, 2, 3] if code in ['20011', '20039'] else [0, 1])]
                statecols = [f for f in statecols if f in features]
                if statecols:
                    bottom = np.zeros(len(clusters))
                    for i, f in enumerate(statecols):
                        vals = gfeatures.groupby('cluster')[f].mean().reindex(clusters).to_numpy()
                        # NaN is visually left blank; no extra filled mass is manufactured.
                        finite = np.isfinite(vals)
                        state_label=f.split('__')[1].replace('_fraction', '')
                        if code=='20101':
                            state_label={'state0':'0:取り上げ','state1':'1:格納'}.get(state_label,state_label)
                        ax.bar(np.flatnonzero(finite), vals[finite]*100, bottom=bottom[finite],
                               label=state_label,
                               color=['#d8dfe5', '#4d8ab7', '#efaa5b', '#69aa89'][i])
                        bottom[finite] += vals[finite]*100
                    ax.legend(fontsize=8)
                ax.set_xticks(range(len(clusters)), [f'C{c}' for c in clusters])
                ax.set_ylim(0, 105)
                ax.set_ylabel('状態比率の症例平均（%）')
                ax.set_title(f'{code} {NAMES[code]}')
            fig.suptitle(model_id + '：状態の構成（初回記録後の保持値）', fontsize=15)
            fig.tight_layout()
            save(fig, f'{slug}_operation_composition.png', '状態の構成', 'model', model_id,
                     '各症例の既知時間を分母にした状態比率を平均しています。初回記録前や対象コード未記録の時間をOFFにしていません。20101は0＝取り上げ、1＝格納です。取り上げは噴射を意味しません。')

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for pos, cl in enumerate(clusters):
                g = joined[joined.cluster == cl]
                axes[0].boxplot([g.duration_s/60], positions=[pos], widths=.55)
                axes[0].scatter(pos + np.linspace(-.14,.14,len(g)),g.duration_s/60,s=9,alpha=.5,color=palette((cl-1)%10))
                sig = audit[audit.case_id.isin(g.case_id)].groupby('case_id').observed_fraction.mean()
                axes[2].boxplot([sig*100], positions=[pos], widths=.55)
            years = sorted(joined.year.unique())
            bottom = np.zeros(len(clusters))
            for i, year in enumerate(years):
                vals = joined.assign(in_year=joined.year.eq(year)).groupby('cluster').in_year.mean().reindex(clusters).to_numpy()*100
                axes[1].bar(range(len(clusters)),vals,bottom=bottom,label=year,color=plt.get_cmap('viridis')(.15+.7*i/max(1,len(years)-1)))
                bottom += vals
            axes[1].legend(fontsize=8)
            for ax in axes:
                ax.set_xticks(range(len(clusters)),[f'C{c}' for c in clusters])
                ax.grid(axis='y',alpha=.15)
            axes[0].set_ylabel('解析区間（分）')
            axes[1].set_ylabel('記録年の構成比（%）')
            axes[2].set_ylabel('10コード平均の観測秒bin率（%）')
            fig.suptitle(model_id+'：クラス差の交絡確認',fontsize=15)
            fig.tight_layout()
            save(fig, f'{slug}_confounds.png', '時間・記録年・観測率の確認', 'model', model_id,
                 'クラスが症例時間や記録年・観測率だけで説明されないかを確認する図です。関連がないことを保証する検定ではありません。')

            # Representative profiles retain one value per observed broad progress bin.
            # No DTW, interpolation, smoothing, or within-bin inventing of observations.
            for cl in clusters:
                reps = representatives[(representatives.model_id == model_id) & (representatives.cluster == cl)].sort_values('rank')
                fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
                handles = []
                for i, rep in enumerate(reps.itertuples()):
                    label = str(rep.case_id).split('__')[-1].replace('Chair_log_seq_', '#')
                    med = bool(rep.is_medoid) if not isinstance(rep.is_medoid,str) else rep.is_medoid.lower()=='true'
                    label += ' [medoid]' if med else ''
                    color = palette(i%10)
                    handles.append(Line2D([],[],color=color,marker='o',linestyle='None',label=label))
                    for ax, code in zip(axes.ravel(), NUMERIC):
                        g = coarse[(coarse.case_id == rep.case_id) & (coarse.code == code)].sort_values('progress_bin')
                        valuecol = 'raw_median' if 'raw_median' in g else 'q50'
                        if valuecol in g:
                            vals = pd.to_numeric(g[valuecol],errors='coerce')
                            good = vals.notna()
                            x = g.loc[good,'progress_bin'].to_numpy(float)
                            # Both zero- and one-based 10-bin contracts are supported.
                            offset = .5 if coarse.progress_bin.min()==0 else -.5
                            ax.scatter((x+offset)*10,vals[good],color=color,s=32 if med else 18,
                                       alpha=.95 if med else .65,marker='D' if med else 'o')
                for ax, code in zip(axes.ravel(),NUMERIC):
                    ax.set_title(f'{code} {NAMES[code]}')
                    ax.set_ylabel('区間内の観測値中央値（記録値）')
                    ax.grid(alpha=.2)
                    ax.set_xlim(0,100)
                    if code in profile_limits:
                        ax.set_ylim(*profile_limits[code])
                    ax.set_xlabel('照明ON→最終OFFの進捗（%）')
                    if not any(len(c.get_offsets()) for c in ax.collections):
                        ax.text(.5,.5,'代表症例で観測なし',transform=ax.transAxes,ha='center',color='#687785')
                fig.suptitle(f'{model_id} / C{cl}：代表症例の大きな段階別プロフィール',fontsize=15,y=.99)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=max(1,len(handles)),fontsize=9)
                fig.tight_layout(rect=[0,0,1,.91])
                save(fig,f'{slug}_C{cl:02d}_representatives.png',f'C{cl} 代表症例の段階別比較','class',model_id,
                     '症例全体を10区間に分けた観測値の中央値です。コードごとの縦軸は全モデル・全クラスで共通です。未観測区間は空白のままです。線で補間せず、DTWで時間を変形していません。')
                fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
                for i, rep in enumerate(reps.itertuples()):
                    med = bool(rep.is_medoid) if not isinstance(rep.is_medoid,str) else rep.is_medoid.lower()=='true'
                    for ax, code in zip(axes.ravel(), STATES):
                        g = coarse[(coarse.case_id == rep.case_id) & (coarse.code == code)].sort_values('progress_bin')
                        if 'active_fraction' in g:
                            vals = pd.to_numeric(g.active_fraction, errors='coerce')
                            good = vals.notna()
                            offset = .5 if coarse.progress_bin.min()==0 else -.5
                            ax.scatter((g.loc[good,'progress_bin'].to_numpy(float)+offset)*10,
                                vals[good]*100,color=palette(i%10),s=38 if med else 22,
                                alpha=.95 if med else .65,marker='D' if med else 'o')
                for ax, code in zip(axes.ravel(), STATES):
                    ax.set_title(f'{code} {NAMES[code]}')
                    ax.set_xlabel('照明ON→最終OFFの進捗（%）')
                    ax.set_ylabel('既知時間に占める格納状態（%）' if code=='20101' else '既知時間に占める非0状態（%）')
                    ax.set_xlim(0,100)
                    ax.set_ylim(-3,103)
                    ax.grid(alpha=.2)
                    if not any(len(c.get_offsets()) for c in ax.collections):
                        ax.text(.5,.5,'代表症例で既知状態なし',transform=ax.transAxes,ha='center',color='#687785')
                fig.suptitle(f'{model_id} / C{cl}：代表症例の大きな段階別・状態構成',fontsize=15,y=.99)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=max(1,len(handles)),fontsize=9)
                fig.tight_layout(rect=[0,0,1,.91])
                save(fig,f'{slug}_C{cl:02d}_representative_states.png',f'C{cl} 代表症例の段階別・状態構成','class',model_id,
                     '10区間ごとに、初回記録後の保持状態が非0であった時間割合を示します。20101の非0は旧特徴での格納状態です（0＝取り上げ、取り上げ≠噴射）。未観測の区間は空白です。非0状態は処置名ではありません。')
            log('model_figures_complete',model_id=model_id,n_cases=len(ass),n_clusters=len(clusters),n_figures=len(figure_index))

        supplemental_figures(out, figure_index, args.dpi, log)
        pd.DataFrame(figure_index).to_csv(out / 'figure_index.csv',index=False)
        build_html(out, manifest, audit, selected, assignments, embedding, figure_index, log)
        log('complete',n_figures=len(figure_index),report=str(out/'report.html'))


def supplemental_figures(out, figures, dpi, log):
    """Late-arriving cross-view and state-coverage audits, without redoing all figures."""
    with tempfile.TemporaryDirectory(prefix='chair-supplement-') as tmp:
        os.environ['MPLCONFIGDIR'] = tmp
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
        def save(fig, name, title, caption):
            path = out/'figures'/name
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            entry = dict(path=str(path.relative_to(out)),title=title,category='overview',model_id='',caption=caption)
            figures[:] = [f for f in figures if f['path'] != entry['path']]
            figures.append(entry)
        crossfile = out/'cross_view_counts.csv'
        if crossfile.exists():
            frame = pd.read_csv(crossfile)
            countcol = next((c for c in ['n_cases','count','n'] if c in frame),None)
            if countcol and len(frame.columns)>=3:
                dimensions = [c for c in frame.columns if c!=countcol]
                if 'operations_cluster' in dimensions:
                    dimensions.remove('operations_cluster')
                    dimensions.insert(0,'operations_cluster')
                matrix = frame.pivot_table(index=dimensions[0],columns=dimensions[1],values=countcol,aggfunc='sum',fill_value=0)
            else:
                matrix = frame.set_index(frame.columns[0]).apply(pd.to_numeric,errors='coerce')
            fig, axes = plt.subplots(1,2,figsize=(12,5))
            values = matrix.to_numpy(float)
            rowtotal = np.nansum(values,axis=1,keepdims=True)
            perc = np.divide(values,rowtotal,out=np.zeros_like(values),where=rowtotal>0)*100
            for ax, vals, title in [(axes[0],values,'症例数'),(axes[1],perc,'各操作クラス内の割合（%）')]:
                im=ax.imshow(vals,aspect='auto',cmap='Blues',vmin=0)
                for i in range(vals.shape[0]):
                    for j in range(vals.shape[1]):
                        ax.text(j,i,f'{vals[i,j]:.0f}'+('%' if ax is axes[1] else ''),ha='center',va='center',
                            color='white' if vals[i,j]>np.nanmax(vals)*.6 else '#243d4d',fontsize=13)
                ax.set_xticks(range(len(matrix.columns)),matrix.columns.astype(str))
                ax.set_yticks(range(len(matrix.index)),matrix.index.astype(str))
                ax.set_xlabel('数値水準のクラス（NO_NUMERICは数値全欠測）')
                ax.set_ylabel('操作構成のクラス')
                ax.set_title(title)
                fig.colorbar(im,ax=ax)
            fig.suptitle('操作構成 × 数値水準：異なる分類軸の組合せ',fontsize=16)
            fig.tight_layout()
            save(fig,'07_cross_view_counts.png','操作構成と数値水準のクロス比較',
                '異なる特徴セットのクラス番号を同一視せず、症例の重なりを直接数えています。複数の軸を組み合わせた記述に使えますが、各セルが独立した処置クラスと検証されたわけではありません。')
        supportfile = out/'support_sensitivity.csv'
        if supportfile.exists():
            support = pd.read_csv(supportfile).drop_duplicates(['model_id','scenario'])
            order = pd.read_csv(out/'selected_models.csv').model_id.tolist()
            matrix = support.pivot(index='model_id',columns='scenario',values='ari').reindex(order)
            scenarios = matrix.columns.tolist()
            fig, ax = plt.subplots(figsize=(10,max(6,len(matrix)*.43)))
            im=ax.imshow(matrix.to_numpy(float),aspect='auto',cmap='RdYlGn',vmin=-.05,vmax=1)
            for i in range(len(matrix)):
                for j in range(len(scenarios)):
                    v=matrix.iloc[i,j]
                    ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=11,
                            color='white' if v<.1 or v>.85 else '#273442')
            labels=[]
            for scenario in scenarios:
                row=support[support.scenario==scenario].iloc[0]
                labels.append(f'数値≧{int(row.min_numeric_bins)}観測bin\n状態既知≧{row.min_state_known_fraction*100:.0f}%')
            ax.set_xticks(range(len(scenarios)),labels)
            ax.set_yticks(range(len(matrix)),matrix.index,fontsize=9)
            ax.set_title('観測支持条件を変えると分類は維持されるか',fontsize=16)
            fig.colorbar(im,ax=ax,label='基準の分類とのARI（内部安定性ARIとは別の比較）')
            fig.tight_layout()
            save(fig,'09_support_sensitivity.png','観測支持条件への感度',
                '少数の数値観測や短い状態既知時間に基づく特徴を欠測扱いにして再比較した感度解析です。緑は基準との一致が高いことを示すだけで、分類の正しさや外部再現性を意味しません。結果が大きく変わるモデルを確定的なクラスとは扱えません。')
        allpath=out/'all_model_comparison_ari_common.csv'
        if allpath.exists():
            matrix=pd.read_csv(allpath,index_col=0)
            fig,ax=plt.subplots(figsize=(max(12,len(matrix)*.63),max(11,len(matrix)*.58)))
            im=ax.imshow(matrix.to_numpy(float),cmap='Blues',vmin=0,vmax=1)
            for i in range(len(matrix)):
                for j in range(len(matrix.columns)):
                    v=matrix.iloc[i,j]
                    ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=7,
                            color='white' if v>.6 else '#213b4b')
            ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=65,ha='right',fontsize=8)
            ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8)
            ax.set_title('全17モデル：同一287症例での分類一致（v1と記録意味反映v2）',fontsize=15)
            fig.colorbar(im,ax=ax,label='モデル間ARI')
            fig.tight_layout()
            save(fig,'10_all_versions_model_agreement.png','v1・v2の全モデルを同一症例で比較',
                '全モデルに共通する287症例に揃えた比較です。クラス番号の直接一致ではなく、症例のまとまり方を比較しています。v2は記録の意味・支持条件・距離を変更した別バージョンです。')
        ass = pd.read_csv(out/'assignments_long.csv')
        selected = pd.read_csv(out/'selected_models.csv')
        ops = selected[(selected.view=='operations') & (selected.method=='kmeans')]
        if len(ops):
            model=str(ops.iloc[0].model_id)
            a=ass[(ass.model_id==model) & ass.cluster.notna()]
            features=pd.read_csv(out/'features.csv').set_index('case_id')
            clusters=sorted(a.cluster.astype(int).unique())
            fig,axes=plt.subplots(2,2,figsize=(13,7))
            for ax,code in zip(axes.ravel(),STATES):
                col=f'E{code}__known_fraction'
                if col in features:
                    for i,cl in enumerate(clusters):
                        ids=a.loc[a.cluster==cl,'case_id']
                        vals=features.reindex(ids)[col].dropna().to_numpy()*100
                        if len(vals):
                            ax.boxplot([vals],positions=[i],widths=.5,showfliers=False)
                            ax.scatter(i+np.linspace(-.15,.15,len(vals)),vals,s=10,color=plt.get_cmap('tab10')((cl-1)%10),alpha=.5)
                        ax.text(i,.98,f'{len(vals)}/{len(ids)}',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
                ax.set_title(f'{code} {NAMES[code]}')
                ax.set_xticks(range(len(clusters)),[f'C{c}' for c in clusters])
                ax.set_ylabel('初回記録後の状態既知時間（%）')
                ax.set_ylim(-3,110)
                ax.grid(axis='y',alpha=.2)
            fig.suptitle(model+'：状態保持の既知時間を確認',fontsize=15)
            fig.tight_layout()
            save(fig,'08_state_known_coverage.png','状態の既知時間と解釈の限界',
                '各状態の初回記録前は未知です。既知時間が短い症例では、既知時間内の状態割合が高くても症例全体の使用割合とは解釈できません。20101=1は格納であり、0の取り上げ状態も噴射時間を意味しません。')
        log('supplemental_figures_updated',cross_view_available=crossfile.exists())


def build_html(out, manifest, audit, selected, assignments, embedding, figures, log):
    h = html.escape
    overview = [f for f in figures if not f['model_id']]
    nclass = int(assignments[assignments.cluster.notna()].case_id.nunique())
    nskip = len(manifest)-nclass
    models = selected.model_id.astype(str).tolist()
    def figure(item):
        return (f'<figure><h3>{h(item["title"])}</h3><a href="{h(item["path"])}" target="_blank">'
                f'<img loading="lazy" src="{h(item["path"])}" alt="{h(item["title"])}"></a>'
                f'<figcaption>{h(item["caption"])}</figcaption></figure>')
    columns = [c for c in ['model_id','view','method','k','n_cases','silhouette','stability_median','stability_q25',
        'stability_q75','stability_min','quality_ari','duration_eta2','date_nmi','valid','evidence_status'] if c in selected]
    table = selected[columns].to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')
    content = ['<!doctype html><html lang="ja"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>参考v1：300症例のDTWなし操作パターン比較</title>',
        '''<style>
        :root{--ink:#1e3343;--muted:#5a6f7d;--line:#dce5eb;--accent:#276d96}
        *{box-sizing:border-box}body{margin:0;color:var(--ink);font:15px/1.75 system-ui,sans-serif;background:#f4f7fa}
        header{background:#163e57;color:white;padding:46px max(24px,calc((100vw - 1360px)/2));}header p{max-width:1000px;color:#deebf3}
        h1{font-size:30px;line-height:1.5;margin:0 0 12px}h2{font-size:24px;margin:14px 0}h3{font-size:18px}
        nav{position:sticky;top:0;background:white;border-bottom:1px solid var(--line);z-index:5;padding:12px 24px;display:flex;gap:20px;flex-wrap:wrap}
        a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}main{max-width:1400px;margin:auto;padding:28px 24px}
        section{padding:24px;background:white;border:1px solid var(--line);border-radius:12px;margin-bottom:26px;scroll-margin-top:80px}
        .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:24px 0}.metric{border:1px solid var(--line);border-radius:8px;padding:18px;background:#f8fbfd}.metric strong{display:block;font-size:29px;color:#275e7f}.metric span{color:var(--muted)}
        figure{margin:28px 0 40px;border-top:1px solid var(--line);padding-top:16px}img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:13px;color:var(--muted);margin-top:10px;max-width:1100px}
        .scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid var(--line);padding:9px;white-space:nowrap;text-align:left}th{background:#edf4f8}
        .note{background:#fff6df;padding:16px 20px;border-left:4px solid #c18b2e}.control{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0}select,input{font:inherit;padding:8px 12px;border:1px solid #aebfc9;border-radius:6px}input{min-width:280px}.model-panel[hidden]{display:none}.muted{color:var(--muted)}
        #scatter{width:100%;height:auto;background:#fbfdff;border:1px solid var(--line);border-radius:8px}#tip{position:fixed;display:none;background:#12384e;color:white;border-radius:6px;padding:10px;font-size:12px;z-index:20;pointer-events:none;max-width:430px;white-space:pre-line}button{font:inherit;background:#276d96;color:white;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}pre{white-space:pre-wrap;font:14px/1.8 system-ui,sans-serif;max-width:1100px}
        @media(max-width:700px){header{padding:28px 20px}h1{font-size:24px}main{padding:15px}section{padding:15px}nav{gap:12px;font-size:13px}}
        </style>''',
        '<header><h1>参考v1：300症例から見る操作パターン<br>DTWを使わない数値・構成・組合せの比較</h1>',
        '<p>照明90012の最初のONから最後のOFFまでを1症例として解析。細かな波形の一致を求めず、数値水準・状態の構成・大きな段階の違いを比較します。</p></header>',
        '<nav><a href="#summary">結果と読み方</a><a href="#cohort">データ監査</a><a href="#comparison">方法の比較</a><a href="#explore">症例を探す</a><a href="#models">モデル別の図</a><a href="#audit">監査ファイル</a></nav>',
        '<main><section id="summary"><h2>今回の範囲</h2><div class="metrics">',
        f'<div class="metric"><strong>{len(manifest)}</strong><span>監査した症例</span></div>',
        f'<div class="metric"><strong>{nclass}</strong><span>分類対象の症例</span></div>',
        f'<div class="metric"><strong>{nskip}</strong><span>対象コード未記録でスキップ</span></div>',
        f'<div class="metric"><strong>{len(models)}</strong><span>比較用の代表モデル</span></div>',
        f'<div class="metric"><strong>{len(figures)}</strong><span>作成した図</span></div></div>',
        '<p class="note"><strong>このページは旧仮定v1の比較記録です。</strong>20776は変更時だけ記録され、次の変更まで持続する設定値です。このv1は記録された1秒binの数値分布を使っており、設定の持続時間を重み付けしていません。設定値保持とシリンジの意味を反映する新版は別ディレクトリ recording_semantics_v2 に保存します。v1の高い内部ARIだけでクラスを確定しません。</p>',
        '<p class="note">クラスはログ上の操作パターン候補です。処置名・診断ではありません。サブサンプリングARIは同じ集団内の安定性であり、未知の症例・別装置・別施設での外部検証ではありません。</p>',
        '<p>数値図は記録された原単位を保持しています。未記録を0やOFFに置き換えていません。状態の持続時間は、初回記録後の値を保持するという仮定に基づきます。20101は0＝取り上げ、1＝格納です。旧v1の非0割合は格納割合を表し、取り上げ状態も実際の噴射時間とは区別します。状態が分かる時間が短い症例もあるため、既知時間の図を併せて確認してください。数値水準モデルでは数値信号が全欠測の症例も分類しないため、モデルごとの分類数が異なる場合があります。</p>']
    reporttxt = out/'report.txt'
    content.append('<h3>比較の入口</h3><ul>')
    for view, title in [('levels','数値水準'),('operations','操作構成'),('combined','数値＋操作の組合せ')]:
        rows = selected[(selected.view==view) & (selected.method=='kmeans')]
        if len(rows):
            r = rows.iloc[0]
            content.append(f'<li><strong>{title}</strong>：{int(r.get("n_cases",0))}症例、k={int(r.k)}、'
                f'内部ARI中央値 {asnum(r.get("stability_median")):.3f}、'
                f'Silhouette {asnum(r.get("silhouette")):.3f}。'
                '原単位のプロフィールと合わせて確認できます。</li>')
    content.append('</ul><p class="muted">内部安定性が高くても、分離度が低い場合は明確に離れた集団ではなく、重なりのある傾向を分けている可能性があります。</p>')
    if (out/'recording_semantics_v2/report.html').exists():
        content.append('<p class="note"><strong>主報告：</strong><a href="recording_semantics_v2/report.html">設定値の持続とシリンジの意味を反映したv2の結果を開く</a>。このページはv1の比較・改善過程の記録です。</p>')
    if (out/'support_sensitivity.csv').exists():
        content.append('<p class="note"><strong>重要な感度解析：</strong>観測点の少ない数値特徴や、状態が分かる時間が短い特徴を除いて再比較すると、一部の分類は大きく変わります。最初の内部ARIが高いことだけでクラスを確定せず、下の「観測支持条件への感度」を必ず確認してください。</p>')
    if reporttxt.exists():
        content.append('<details><summary>解析担当の評価と判断を読む（全文）</summary><pre>'+h(reporttxt.read_text())+'</pre></details>')
    summary = out/'evaluation_summary.json'
    if summary.exists():
        content.append('<details><summary>評価の構造化記録</summary><pre>'+h(summary.read_text())+'</pre></details>')
    content.extend(['</section><section id="cohort"><h2>データの記録状況</h2>'])
    content.extend(figure(f) for f in overview if f['path'].split('/')[-1][:2] in ['01','02','03','08'])
    content.extend(['</section><section id="comparison"><h2>方法を同じ条件で比べる</h2>',
        '<p>クラス数・分離度・安定性・クラスの大きさ・観測状況との関連を合わせて確認します。モデルごとのC1、C2などの番号は同一のクラスを意味しません。モデル間の一致はARIで比較します。</p>',
        '<div class="scroll">'+table+'</div>'])
    content.extend(figure(f) for f in overview if f['path'].split('/')[-1][:2] not in ['01','02','03','08'])
    content.extend(['</section><section id="explore"><h2>モデルを切り替えて症例を見る</h2>',
        '<p>下のPCA図は拡大解釈せず、症例の位置を探索するために使ってください。点にカーソルを重ねると症例ID・クラス・記録日・時間が表示されます。</p>',
        '<div class="control"><label>モデル <select id="scatterModel"></select></label><label>症例検索 <input id="caseSearch" placeholder="例：1558、2024-06、症例ID"></label><button id="reset">検索を解除</button></div>',
        '<svg id="scatter" viewBox="0 0 1100 600" role="img" aria-label="症例のPCA投影"></svg><p id="scatterStatus" class="muted"></p>',
        '</section><section id="models"><h2>モデル別・クラス別の図</h2>',
        '<div class="control"><label>表示モデル <select id="modelSelect"></select></label></div>'])
    for model in models:
        row = selected[selected.model_id==model].iloc[0]
        content.append(f'<div class="model-panel" data-model="{h(model)}"><h2>{h(model)}</h2>')
        content.append(f'<p>{h(VIEWS.get(str(row.get("view")),str(row.get("view"))))} / {h(str(row.get("method")))} / k={int(row.get("k"))}</p>')
        content.extend(figure(f) for f in figures if f['model_id']==model)
        content.append('</div>')
    content.extend(['</section><section id="audit"><h2>監査・再現用ファイル</h2><p>入力症例の一覧、特徴定義、全候補の比較、各症例の分類、代表症例、安定性反復、判断過程を確認できます。</p><ul>'])
    for name in ['report.txt','evaluation_summary.json','case_manifest.csv','boundary_audit.csv','signal_audit.csv',
        'feature_catalog.csv','features.csv','model_candidates.csv','selected_models.csv','assignments_long.csv',
        'class_profiles.csv','representatives.csv','stability_repeats.csv','model_comparison_ari.csv',
        'model_comparison_ari_common_cohort.csv','pairwise_model_comparison_n.csv',
        'all_model_comparison_ari_common.csv','case_assignments_v2_candidate.csv',
        'cross_view_counts.csv','class_quality_audit.csv','sensitivity_summary.csv',
        'support_sensitivity.csv',
        'figure_index.csv','extraction_metadata.json','analysis_metadata.json']:
        if (out/name).exists():
            content.append(f'<li><a href="{name}">{name}</a></li>')
    for file in sorted((out/'process').glob('*')):
        if file.is_file():
            content.append(f'<li><a href="process/{h(file.name)}">process/{h(file.name)}</a></li>')
    content.append('</ul></section></main><div id="tip"></div>')
    datapoints = {}
    mi = manifest.set_index('case_id')
    for model in models:
        emb = embedding[embedding.model_id==model].merge(assignments[assignments.model_id==model][['case_id','cluster']],on='case_id',validate='one_to_one')
        records = []
        for r in emb.itertuples():
            if not np.isfinite(asnum(getattr(r,'pc1',np.nan))) or not np.isfinite(asnum(getattr(r,'pc2',np.nan))):
                continue
            m = mi.loc[r.case_id]
            records.append(dict(id=r.case_id,x=float(r.pc1),y=float(r.pc2),c=int(r.cluster),
                                date=str(m.get('date','')),minutes=round(float(m.duration_s)/60,2)))
        datapoints[model] = records
    content.append('<script>const points='+json.dumps(datapoints,ensure_ascii=False).replace('</','<\/')+';')
    content.append('''
    const models=Object.keys(points),colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
    const svg=document.getElementById('scatter'),tip=document.getElementById('tip'),search=document.getElementById('caseSearch');
    for(const id of ['scatterModel','modelSelect']){const el=document.getElementById(id);models.forEach(m=>{const opt=document.createElement('option');opt.value=m;opt.textContent=m;el.appendChild(opt);});}
    function node(tag,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text)el.textContent=text;svg.appendChild(el);return el;}
    function draw(){const m=document.getElementById('scatterModel').value,data=points[m]||[],q=search.value.toLowerCase();svg.replaceChildren();if(!data.length)return;
      const xs=data.map(d=>d.x),ys=data.map(d=>d.y),minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys),sx=x=>75+(x-minx)/(maxx-minx||1)*850,sy=y=>530-(y-miny)/(maxy-miny||1)*450;
      node('line',{x1:65,y1:540,x2:950,y2:540,stroke:'#8ba2b0'});node('line',{x1:65,y1:540,x2:65,y2:60,stroke:'#8ba2b0'});node('text',{x:470,y:580,fill:'#526d7f','font-size':15},'PC1');node('text',{x:15,y:280,fill:'#526d7f','font-size':15},'PC2');
      let found=0;data.forEach(d=>{const hit=!q||d.id.toLowerCase().includes(q);if(hit)found++;const c=node('circle',{cx:sx(d.x),cy:sy(d.y),r:q&&hit?7:4.5,fill:colors[(d.c-1)%colors.length],opacity:hit ? 0.83 : 0.07,stroke:q&&hit?'#102f42':'none','stroke-width':1.5});c.addEventListener('mouseenter',e=>{tip.textContent=d.id+'\\nクラス C'+d.c+' / '+d.date+' / '+d.minutes+'分';tip.style.display='block';});c.addEventListener('mousemove',e=>{tip.style.left=Math.max(8,Math.min(e.clientX+14,window.innerWidth-440))+'px';tip.style.top=(e.clientY+12)+'px';});c.addEventListener('mouseleave',()=>tip.style.display='none');});
      [...new Set(data.map(d=>d.c))].sort((a,b)=>a-b).forEach((c,i)=>{node('circle',{cx:980,cy:90+i*28,r:5,fill:colors[(c-1)%colors.length]});node('text',{x:994,y:95+i*28,fill:'#365063','font-size':13},'C'+c+' ('+data.filter(d=>d.c===c).length+')');});
      document.getElementById('scatterStatus').textContent=q?'検索一致 '+found+' / '+data.length+' 症例':'このモデルで表示・分類した '+data.length+' 症例。対象10コード未記録、またはこの特徴セットに観測値がない症例は、割当せず監査に保持しています。';}
    function panel(){const model=document.getElementById('modelSelect').value;document.querySelectorAll('.model-panel').forEach(el=>el.hidden=el.dataset.model!==model);}
    document.getElementById('scatterModel').addEventListener('change',draw);document.getElementById('modelSelect').addEventListener('change',panel);search.addEventListener('input',draw);document.getElementById('reset').addEventListener('click',()=>{search.value='';draw();});draw();panel();
    </script></html>''')
    (out/'report.html').write_text('\n'.join(content),encoding='utf-8')
    log('html_report_written',bytes=(out/'report.html').stat().st_size,interactive_points=sum(map(len,datapoints.values())))


if __name__ == '__main__':
    main()
