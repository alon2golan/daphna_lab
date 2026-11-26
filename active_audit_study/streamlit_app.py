"""Explore matched signed graph audits on CIFAR-100."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).with_name('benchmark_metrics.csv')
COHORTS = {
    'coverage': 'Seeded 13-neighbor coverage',
    'maxherding': 'Archived MaxHerding order',
}
CONDITIONS = {
    'clean': 'Clean',
    'sym30': 'Symmetric 30%',
    'sym40': 'Symmetric 40%',
    'sym50': 'Symmetric 50%',
    'asym30': 'Asymmetric 30%',
    'asym40': 'Asymmetric 40%',
    'asym50': 'Asymmetric 50%',
    'cifar100n': 'CIFAR-100N',
}
POLICIES = {
    'signed_peer': 'Signed graph audits',
    'loo_audit': 'Leave-one-out audits',
    'fresh_clean': 'Buy new clean labels',
    'single_footprint': 'Unsigned impact',
    'diverse_footprint': 'Diverse unsigned impact',
    'random_audit': 'Random audits',
    'no_audit': 'Initial labels only',
}
METRICS = {'test_correct': 'Graph classifier', 'test_1nn_correct': 'Cosine 1NN'}


@st.cache_data(show_spinner=False)
def load_data(path, modified):
    return pd.read_csv(path)


def pair_scores(data, metric, method, reference):
    scores = data.assign(accuracy=100 * data[metric] / data.test_count).pivot(
        index=['condition', 'budget', 'seed'], columns='policy', values='accuracy',
    )
    pairs = scores[[method, reference]].copy()
    if pairs.isna().any().any():
        raise ValueError('Some evaluations are missing a matched reference.')
    pairs.columns = ['method', 'reference']
    pairs['delta'] = (pairs.method - pairs.reference).round(8)
    pairs['win'] = pairs.delta.gt(0)
    pairs['tie'] = pairs.delta.eq(0)
    summary = pairs.groupby(level=['condition', 'budget']).agg(
        method=('method', 'mean'), reference=('reference', 'mean'),
        delta=('delta', 'mean'), wins=('win', 'sum'), ties=('tie', 'sum'),
        seeds=('delta', 'size'),
    )
    return pairs, summary


def heatmap(summary, reference):
    conditions = [c for c in CONDITIONS if c in summary.index.get_level_values('condition')]
    budgets = sorted(summary.index.get_level_values('budget').unique())
    matrix = summary.delta.unstack().loc[conditions, budgets]
    wins = summary.wins.unstack().loc[conditions, budgets]
    total = summary.seeds.unstack().loc[conditions, budgets]
    hover = np.stack([wins.to_numpy(), total.to_numpy()], axis=-1)
    span = max(1., float(np.abs(matrix.to_numpy()).max()))
    figure = go.Figure(go.Heatmap(
        x=[f'{b:,}' for b in budgets], y=[CONDITIONS[c] for c in conditions],
        z=matrix.to_numpy(), zmin=-span, zmax=span,
        colorscale=[[0, '#e48c75'], [.5, '#f8f7f4'], [1, '#5dbdac']],
        xgap=4, ygap=4, texttemplate='%{z:+.2f}',
        textfont=dict(color='#172a37', size=14), customdata=hover,
        colorbar=dict(title='pp', thickness=12),
        hovertemplate=(
            '%{y} · %{x} initial labels<br>%{z:+.2f} pp'
            '<br>%{customdata[0]:.0f}/%{customdata[1]:.0f} seed wins<extra></extra>'
        ),
    ))
    figure.update_layout(
        height=115 + 40 * len(conditions), margin=dict(t=10, b=40, l=3, r=0),
        xaxis=dict(title='Initial labels', side='bottom', type='category', automargin=True),
        yaxis=dict(autorange='reversed', automargin=True),
    )
    st.plotly_chart(figure, config={'displaylogo': False, 'displayModeBar': False},
                    alt=f'Paired mean accuracy differences against {POLICIES[reference]}.')


def render():
    st.set_page_config(page_title='Signed graph audits', layout='wide')
    st.title('Signed graph audits')
    st.caption('CIFAR-100 BYOL · 10,000 held-out images · extra clean requests equal to 10% of initial labels')
    if not DATA.is_file():
        st.info('The verification table appears after the matched runs finish.')
        st.stop()
    data = load_data(DATA, DATA.stat().st_mtime_ns)
    controls = st.columns([1.25, 1, 1.25])
    cohort = controls[0].selectbox('Acquisition order', COHORTS, format_func=COHORTS.get)
    metric = controls[1].selectbox('Test classifier', METRICS, format_func=METRICS.get)
    run = data[data.acquisition.eq(cohort)]
    known = set(run.policy)
    method = 'signed_peer'
    choices = [p for p in POLICIES if p != method and p in known]
    reference = controls[2].selectbox('Reference', choices, format_func=POLICIES.get)
    if cohort == 'maxherding':
        st.caption('Shared archived MaxHerding order. Seeds redraw the synthetic noise.')
    else:
        st.caption('Each seed regenerates the acquisition order and synthetic corruption. CIFAR-100N uses fixed human labels.')

    pairs, summary = pair_scores(run, metric, method, reference)
    synthetic = summary.loc[[c for c in CONDITIONS if c.startswith(('sym', 'asym'))]]
    indicators = st.columns(3)
    indicators[0].metric('Synthetic cells with a gain', f'{synthetic.delta.gt(0).sum()} / {len(synthetic)}')
    indicators[1].metric('Cells with a majority of seed wins', f'{(2 * synthetic.wins > synthetic.seeds).sum()} / {len(synthetic)}')
    indicators[2].metric('Median synthetic gain', f'{synthetic.delta.median():+.2f} pp')
    st.subheader(f'Accuracy change vs {POLICIES[reference].lower()}')
    heatmap(summary, reference)

    st.subheader('Scores by budget')
    condition = st.selectbox('Noise setting', [c for c in CONDITIONS if c in set(run.condition)],
                             index=1 if cohort == 'coverage' else 0, format_func=CONDITIONS.get)
    detail = summary.loc[condition].reset_index()
    detail['wins'] = detail.wins.astype(str) + ' / ' + detail.seeds.astype(str)
    st.dataframe(
        detail[['budget', 'method', 'reference', 'delta', 'wins', 'ties']],
        hide_index=True,
        column_config={
            'budget': 'Initial labels',
            'method': st.column_config.NumberColumn('Signed audits (%)', format='%.2f'),
            'reference': st.column_config.NumberColumn(POLICIES[reference] + ' (%)', format='%.2f'),
            'delta': st.column_config.NumberColumn('Difference (pp)', format='%+.2f'),
            'wins': 'Seed wins',
            'ties': 'Ties',
        },
    )
    with st.expander('Paired scores by seed'):
        st.dataframe(pairs.loc[condition].drop(columns=['win', 'tie']).reset_index().round(2), hide_index=True)
    with st.expander('Method and protocol'):
        st.markdown(
            'Leave-one-out scores the gap between the observed label and its strongest rival, '
            'using only other annotations. Signed auditing simulates that switch, sums the change '
            'in peer support over the affected predictions, and multiplies by the positive gap. '
            'Two flips worth +0.5 and −0.4 with a gap of 0.6 yield a score of 0.06.'
        )
        st.caption(
            'Clean replies are simulated from CIFAR-100 ground truth. Audit rankings use the training graph. '
            'Budgets within a seed share an acquisition prefix. The classifiers share the same held-out test set.'
        )
    st.download_button('Download matched seed scores', DATA.read_bytes(), file_name=DATA.name, mime='text/csv')


render()
