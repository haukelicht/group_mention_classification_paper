#!/usr/bin/env python
# coding: utf-8

# Setup



import os
from pathlib import Path
import gc

import pandas as pd
import numpy as np
import pickle

from scipy.stats import chi2_contingency

import re
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.fighting_words import compute_fighting_words, lemma_tokenizer, ENGLISH_STOP_WORDS
stopwords = list(ENGLISH_STOP_WORDS)

from src.reporting import *
from tqdm.auto import tqdm

from matplotlib.path import Path as PatchPath
import matplotlib.patches as patches




base_path = Path(__file__).resolve().parent / Path('..')
data_path = base_path / 'data'
labeled_path = data_path / 'labeled'
manifestos_path = data_path / 'manifestos'
annotations_path = data_path / 'annotations'

results_path = base_path / 'results'
intermediate_path = results_path / 'intermediate'
figures_path = results_path / 'figures'
tables_path = results_path / 'tables'


# Dataset Descriptives



# fp = data_path / 'labeled_mentions_with_party_metadata.pkl'
# df = pd.read_pickle(fp)
fp = data_path / 'labeled_mentions_with_party_metadata.feather'
df = pd.read_feather(fp)




fp = manifestos_path / 'all_manifesto_sentences.tsv'
cols = ['country_iso3c', 'party_id', 'date', 'party_family', 'manifesto_id', 'sentence_id', 'lang']
sentences_df = pd.read_csv(fp, sep='\t', usecols=cols)




# FIGURE A1
fig_cap = "Number of manifestos by party family (all years)."

count_manifestos = df.groupby('party_family', observed=False).manifesto_id.nunique()
count_manifestos = count_manifestos[count_manifestos>0].sort_index()
count_manifestos.index = count_manifestos.index.map(family_map)

fig = count_manifestos.plot(
    kind='barh', 
    color=[all_fam_palette[f] for f in count_manifestos.index], 
    figsize=(6, 2), 
    xlabel='Number of manifestos', 
    ylabel=""
)
# plt.show()

save_figure(fig, 'figureA01.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE A2
fig_cap = "Number of sentences by party family (all years)."

count_sentences = df.groupby('party_family', observed=False).sentence_id.nunique()
count_sentences = count_sentences[count_sentences>0].sort_index()
count_sentences.index = count_sentences.index.map(family_map)

fig = count_sentences.plot(
    kind='barh', 
    color=[all_fam_palette[f] for f in count_sentences.index], 
    figsize=(6, 2), 
    xlabel='Number of sentences', 
    ylabel=""
)
# add comma at thousands separator for x-axis
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
# plt.show()

save_figure(fig, 'figureA02.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE A3
fig_cap = "Overview of cases included in our corpus. Each square represents a party in a given year, colored by its party family."

tmp = df[['country_iso3c', 'year', 'party_name', 'party_family']].drop_duplicates()
tmp.party_family = tmp.party_family.map(family_map)

countries = tmp['country_iso3c'].unique().tolist()

heights = tmp.groupby('country_iso3c').aggregate({'party_name': 'nunique'})['party_name']

n_col = 1
fig, axes = plt.subplots(
    len(countries)//n_col, n_col, 
    figsize=(5, heights.sum()/4), 
    height_ratios=heights.to_list(),
    sharex=True, 
    gridspec_kw={'hspace': 0.4}
    )

axes = axes.flatten()

scatter_kwargs = dict(
    x='year', 
    y='party_name', 
    hue='party_family', 
    marker='s', 
    palette=all_fam_palette,
    s=100, 
    legend=False
)
for i, (ctr, subdf) in enumerate(tmp.groupby("country_iso3c")):
    ax = axes[i]
    sns.scatterplot(data=subdf, ax=ax, **scatter_kwargs)
    # ax.set_ylim(1.15, -0.15)
    ax.set_ylabel(None)
    # add country label as y-axis label (second axis) on right hand side
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(None)
        spine.set_visible(False)
    ax_right = ax.twinx()
    ax_right.set_ylabel(ctr, fontweight='bold', rotation=0, labelpad=15)
    ax_right.set_yticks([])
    # ax_right.spines['right'].set_visible(False)
    # increase y-limits so that squares are fully visible
    ax.set_ylim(-0.5, len(subdf['party_name'].unique()) - 0.5)
    # make plot backgorund light gray
    ax.set_facecolor('#f0f0f0')

save_figure(fig, 'figureA03.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE A4
fig_cap = "Language coverage of the corpus of manifesto sentences. The corpus includes manifestos in 26 different languages. Languages denoted with their ISO 639-1 code. German (de), English (en), and Swedish (sv) are more strongly represented because our corpus also includes extensive time-series for social democratic and conservative parties in Germany, Sweden, the United Kingdom, and the United States. Other factors that contribute to languages' varying prevalence in our corpus are party system fragmentation and manifesto length, which, for example, explains the high number of Dutch (nl) sentences (many parties imply many manifestos)."

fig = sentences_df.lang.value_counts().plot(kind='bar', figsize=(6, 1.5), color='black')
# rotate x-axis labels to horizontal
plt.xticks(rotation=0)
# format label of y-axis in thousands with comma separator
# plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.0f}".format(x)))
plt.xlabel('Language')
# plt.show()

save_figure(fig, 'figureA04.pdf', caption=fig_cap, dest=figures_path) 




most_common = df.text.value_counts().head(6).apply(lambda x: f"{x:,}")
most_common = [f'"{m}" (N={n})' for m, n in zip(most_common.index, most_common.values)]

# print(', '.join(most_common[:-1]) + f', and {most_common[-1]}')




# FIGURE 1
fig_cap = r"Share of manifesto sentences mentioning at least one social group, by party family based on social group mentions extracted with our fine-tuned RoBERTa model. The boxplots show the distribution across manifestos within each party family, excluding outliers."

tmp = pd.merge(
    df.groupby(['party_id', 'date'], observed=False).agg(
        n_mentions=('party_id', 'count'), 
        n_binary=('sentence_id', 'nunique')
    ).reset_index(),
    sentences_df.groupby(['party_id', 'date', 'party_family'], observed=False).size().reset_index(name='n_sentences'),
    on=['party_id', 'date']
)
tmp['mention_to_sentence_ratio'] = tmp['n_mentions'] / tmp['n_sentences']
tmp['mention_prevalence'] = tmp['n_binary'] / tmp['n_sentences']
tmp = tmp.query("party_family != 'other'")
tmp['party_family'] = tmp['party_family'].apply(normalize_family).astype('category').cat.set_categories(all_fam_palette.keys())

# plot boxplots of mention_to_sentence_ratio by party_family

plt.figure(figsize=(6.5, 1.8))
fig = sns.boxplot(
    data=tmp, 
    y='party_family', 
    hue='party_family',
    palette=all_fam_palette, 
    x='mention_prevalence', 
    showfliers=False
)
plt.xlim(0, 1)
plt.ylabel(None)
plt.xlabel('share of manifesto sentences mentioning at least one social group')
# plt.show()

save_figure(fig, 'figure01.pdf', caption=fig_cap, dest=figures_path) 


# Annotation

## Mention detection



fp = data_path / "annotations/group_mention_detection/consolidated_annotations.jsonl"
fp = annotations_path / "group_mention_detection" / "consolidated.jsonl"
annos = pd.read_json(fp, lines=True)

meta = annos["metadata"].apply(pd.Series)
annos[meta.columns] = meta

annos = annos.merge(sentences_df[["sentence_id", "party_family"]], on="sentence_id", how="left")

annos["year"] = (annos['date'].astype(int).round(-2)/100).astype(int)




# FIGURE B1
fig_cap = "Distribution of sentences annotated for group mention detection over time."

plt.figure(figsize=(6, 3))
fig = annos.year.plot(kind='hist', bins=range(1965, 2025, 5), rwidth=0.8, color='grey')
plt.ylabel('Number of annotated sentences')
plt.xlabel('Year')
# plt.show()

save_figure(fig, 'figureB01.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE B2
fig_cap = "Distribution of sentences annotated for group mention detection across countries."

fig = (
    annos.
    groupby("country_iso3c").
    size().
    reset_index(name='n_sentences').
    sort_values('n_sentences', ascending=True).
    plot(
        x="country_iso3c", 
        y="n_sentences", 
        kind='barh',
        color='grey', 
        figsize=(4, 6),
        legend=False
    )
)
plt.xlabel('Number of annotated sentences')
plt.ylabel('Country')
# plt.show()
save_figure(fig, 'figureB02.pdf', caption=fig_cap, dest=figures_path) 


## Attribute classification



## report reliability by round

# TODO: add ICA estimates here

annoation_rounds = {
    '1': 'social-group-mention-categorization-round01',
    '2': 'social-group-mention-categorization-round02',
    '3': 'social-group-mention-categorization-round03',
}

folder = annotations_path / 'group_mention_categorization' / 'rounds' 
ica_raw = pd.concat({
    r: pd.read_pickle(folder / f / 'parsed' / 'ica_estimates.pkl') 
    for r, f in annoation_rounds.items()
})
ica_raw.reset_index(level=0, names=['round'], inplace=True)
ica_raw.loc[ica_raw.label.isna(), 'label'] = ica_raw.loc[ica_raw.label.isna(), 'q_category']
ica_raw.loc[ica_raw.q_id=='universal_attributes', 'label'] = 'overall'

ica = ica_raw.loc[ica_raw.q_id.str.endswith('_attributes'), ['round', 'q_id', 'label', 'prop_yes', 'krippendorff_alpha']]
ica = ica[~ica.label.isna()]

cats_map = {i.split('__')[1]: nm for i, nm in attribute_category_names_map.items()}
ica['label'] = ica.label.replace({"education level": "education"})
ica['category'] = ica.label.replace(cats_map)




# TABLE C1
tbl_cap = r"Inter-coder agreement estimates for attribute classifications computed at the level of attribute dimensions: universal, economic, and non-economic. Estimates are based on Krippendorff's $\alpha$ and the prevalence of `yes' annotations (prevalence) across all annotated examples in each annotation round."

ica_overall = ica.loc[ica.label == 'overall', ['round', 'q_id', 'prop_yes', 'krippendorff_alpha']]
ica_overall = ica_overall.rename(columns={'prop_yes': 'prevalence', 'krippendorff_alpha': r"Krippendorff's $\alpha$"})
ica_overall['q_id'] = ica_overall.q_id.str.removesuffix('_attributes')
ica_overall['q_id'] = pd.Categorical(ica_overall['q_id'], categories=['universal', 'economic', 'non-economic'], ordered=True)
ica_overall = ica_overall.pivot_table(index='q_id', columns=['round'], observed=True).round(3)
ica_overall.index.name = None
ica_overall.columns.names = [None, 'annotation round']

tab = format_table(ica_overall, caption=tbl_cap, index=True, label='tableC01')
save_table(tab, 'tableC01.tex', dest=tables_path)




# TABLE C2
tbl_cap = r"Inter-coder agreement estimates for attribute classifications computed at the level of economic and non-economic attribute categories. Estimates are based on Krippendorff's $\alpha$ and values in parentheses indicate the prevalence of `yes' annotations (prevalence) across all annotated examples in each annotation round."

ica_cats = ica.loc[ica.label != 'overall', ['round', 'q_id', 'category', 'prop_yes', 'krippendorff_alpha']]
ica_cats['category'] = pd.Categorical(ica_cats['category'], categories=list(cats_map.values()), ordered=True)
ica_cats['value'] = ica_cats.apply(lambda r: rf"{r['krippendorff_alpha']:+0.3f} ({r['prop_yes']*100:0.1f}\%)", axis=1)
ica_cats['q_id'] = ica_cats.q_id.str.removesuffix('_attributes')
ica_cats = ica_cats.pivot_table(index=['q_id', 'category'], columns=['round'], values=['value'], aggfunc='first', observed=False).round(3).fillna('').sort_index()
ica_cats.columns = pd.Index(ica_cats.columns.get_level_values(1), name='annotation round')
ica_cats.index.names = ['dimension', 'category']

tab = format_table(ica_cats, caption=tbl_cap, index=True, resize=True, label='tableC02')
save_table(tab, 'tableC02.tex', dest=tables_path) 


# Classifier evaluation

## Reliability



splits_path = data_path /'data_splits' / 'group_mention_categorization'

tmp = pd.concat({
    split_file.parent.name: pd.read_pickle(split_file)
    for split_file in splits_path.glob("**/train.pkl")
}).reset_index(level=0, names=['fold'])
tmp.rename(columns={'economic__education_level': 'economic__education'}, inplace=True)
train_set_prevalences = tmp[label_cols].mean().astype(float).rename(index=attribute_category_names_map)
train_set_prevalences = train_set_prevalences.to_frame('prevalence').reset_index(names=['category'])




model = "sentence-transformers--all-mpnet-base-v2"
tasks = [
    "economic_attributes_classification",
    "noneconomic_attributes_classification"
]

discard_these = ["samples avg"] 
metrics = ['f1-score', 'precision', 'recall']

idx_map = {
    'micro avg': 'micro average', 
    'weighted avg': 'weighted average',
    'macro avg': 'macro average', 
    **attribute_category_names_map
}

eval_res = {}
eval_res_tabs = {}

for task in tasks:
    results_dir = results_path / 'classifier_evaluation' / "hp_search" / task
    res = pd.concat({
        fp.parts[-5:-1]: pd.read_json(fp).T.reset_index(level=0, names="what") 
        for fp in results_dir.glob("**/eval_results.json")
    })
    res.reset_index(level=[0,1,2,3], names=["method", "model_name", "strategy", "fold"], inplace=True)
    res = res.query("what not in @discard_these").copy()
    res["what"] = res["what"].replace({'economic__education_level': 'economic__education'})
    res["category"] = res["what"].replace(idx_map)
    eval_res[task] = res

    res_tab = res.groupby("category", observed=True)[metrics].mean()
    res_tab.columns = res_tab.columns.str.title()
    res_tab = res_tab.reset_index().merge(train_set_prevalences, on="category", how="left").round(3)
    res_tab['category'] = pd.Categorical(res_tab['category'], categories=list(idx_map.values()), ordered=True)
    res_tab = res_tab.sort_values('category', ascending=True)
    res_tab['category'] = res_tab['category'].astype(str).replace({v: rf"\quad \textit{{{v}}}" for v in attribute_category_names_map.values()})

    eval_res_tabs[task] = res_tab




# TABLE 2
tbl_cap = r"Evaluation results of group mention attribute classifiers. Values report the mean precision, recall, and F1-score averaged across five folds for each attribute category, along with the prevalence of each category in the training set."

tab = pd.concat([
    pd.DataFrame({"category": r"\textbf{economic attributes}"}, index=[0]),
    eval_res_tabs[tasks[0]],
    pd.DataFrame({"category": r"\textbf{non-economic attributes}"}, index=[0]),
    eval_res_tabs[tasks[1]],
])
tab.set_index('category', inplace=True)
tab = tab.replace(np.nan, '')
tab.index.name = None

tab = format_table(tab, caption=tbl_cap, escape=False, index=True, multicolumn_cmidrules=[(10, 0, 4)], label='table02')
save_table(tab, 'table02.tex', dest=tables_path) 


## Validation

### "universal" mentions



vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=stopwords,
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)
analyzer = vectorizer.build_analyzer()

# Split data into universal (neither) vs. any attributes
df["is_universal"] = ~df[label_cols].any(axis=1)
df_universal = df[["text", "is_universal"]].copy()
del df["is_universal"]

fw_universal = compute_fighting_words(
    l1=df_universal.loc[ df_universal["is_universal"], 'text'].tolist(),  # universal mentions
    l2=df_universal.loc[~df_universal["is_universal"], 'text'].tolist(),  # mentions with any attributes
    cv=vectorizer,
)

fw_universal = pd.DataFrame(fw_universal, columns=['word', 'score']).sort_values('score', ascending=False)




# FIGURE C1
fig_cap = r"Most distinctive words for mentions with no specific attributes (\emph{universal} mentions, left) vs. mentions with at least one economic or non-economic attribute (right).  Values plotted are $z$-scores from ``fighting words'' analysis.  Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

# Get top 20 lowest (most negative) and highest (most positive) scores
top_negative = fw_universal.nsmallest(20, 'score').sort_values('score', ascending=False)
top_positive = fw_universal.nlargest(20, 'score').sort_values('score', ascending=True)

# Create two-column layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

# Left plot: positive scores (distinctive for universal mentions)
ttl = r"universal mentions (no specific attributes)"
ax1.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax1.barh(range(len(top_positive)), top_positive['score'], color='#1b9e77', zorder=2)
ax1.set_yticks(range(len(top_positive)))
ax1.set_yticklabels(top_positive['word'])
ax1.set_xlabel('z-score', fontsize=11)
ax1.set_title(ttl, fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
plt.setp(ax1.get_yticklabels(), ha='left')
ax1.set_xlim(0, 120)
ax1.invert_xaxis()  # Invert to show positive values extending left

# Right plot: negative scores (distinctive for mentions with attributes)
ttl = r"mentions with specific attributes"
ax2.axvline(x=-1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax2.barh(range(len(top_negative)), top_negative['score'], color='#d95f02', zorder=2)
ax2.set_yticks(range(len(top_negative)))
ax2.set_yticklabels(top_negative['word'])
ax2.set_xlabel('z-score', fontsize=11)
ax2.set_title(ttl, fontweight='bold', fontsize=12)
ax2.set_xlim(-120, 0)
ax2.invert_xaxis()  # Invert to show negative values extending left

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureC01.pdf', caption=fig_cap, dest=figures_path) 




# TABLE C5
# tbl_cap = r"Examples of \emph{universal} group references.  Values computed by summing ``fighting words'' scores as weights of mentions' tokens, normalized by number of tokens."

# fw_lookup = {r['word']: r['score'] for r in fw_universal.to_dict(orient='records')}
# fw_vals = np.array([fw_lookup[f] for f in vectorizer.get_feature_names_out()])
# analyzer = vectorizer.build_analyzer()

# vectorize mentions 
# universal_mentions = df_universal[df_universal['is_universal']].reset_index(drop=True)
# mentions_texts = universal_mentions['text']

# X_mentions = vectorizer.transform(mentions_texts.tolist())
# binarize
# X_mentions[X_mentions>0] = 1
# apply z-score values to each row in `X_mentions` as weights
# X_mentions_scores = X_mentions @ fw_vals[:, np.newaxis]
# normalize for mention length
# X_mentions_scores /= X_mentions.sum(axis=1)

# universal_mentions['score'] = X_mentions_scores[:, 0]
# rank = universal_mentions['score'].argsort()[::-1]

# tab = universal_mentions[universal_mentions['score']>1.96].copy()
# tab['text_norm'] = tab['text'].apply(lambda x: ' '.join(analyzer(x)).strip())
# tab = tab.drop_duplicates('text_norm').sort_values('score', ascending=False).head(400)

# n_ = 20
# tab = tab.sample(n_, weights=tab['score'].abs()**2, random_state=42)

# tab = tab[['text', 'score']].sort_values('score', ascending=False).reset_index(drop=True)
# tab.columns = ["Mention", "$z$-score"]

# tab = format_table(tab)
# save_table(tab, 'tableC05.tex', dest=tables_path) 


### Comparison to Thau's group mention categorizations



# Split data into universal (neither) vs. any attributes
df["universal"] = ~df[label_cols].any(axis=1)




df_universal = df.query("~universal").copy()

attribute_cats_fws_fp = intermediate_path / 'attribute_cats_fws.pkl'
attribute_cats_examples_fp = intermediate_path / 'attribute_cats_examples.pkl'
attribute_fw_topterm_performances_fp = intermediate_path / 'attribute_fw_topterm_performances.pkl'
UPDATE = False

if UPDATE or not attribute_cats_fws_fp.exists():
    attribute_cats_fws_fp.parent.mkdir(exist_ok=True)

    vectorizer = CountVectorizer(
        tokenizer=lemma_tokenizer,
        token_pattern=None,
        lowercase=False,  # handled in tokenizer
        stop_words=stopwords,
        ngram_range=(1, 3), 
        max_df=0.8,
        min_df=5
    )
    analyzer = vectorizer.build_analyzer()

    attribute_cats_fws = {}
    attribute_cats_examples = {}
    for lab in tqdm(label_cols):
        fw = compute_fighting_words(
            l1=df_universal.loc[df_universal[lab]==1, 'text'].tolist(),
            l2=df_universal.loc[df_universal[lab]==0, 'text'].tolist(),
            cv=vectorizer,
        )

        fw = pd.DataFrame(fw, columns=['word', 'score'])
        attribute_cats_fws[lab] = fw.nlargest(50, 'score').sort_values('score', ascending=False)

        fw_lookup = {r['word']: r['score'] for r in fw.to_dict(orient='records')}
        fw_vals = np.array([fw_lookup[f] for f in vectorizer.get_feature_names_out()])

        # vectorize mentions 
        focal_mentions = df_universal[df_universal[lab]==1].reset_index(drop=True)
        mentions_texts = focal_mentions['text']

        X_mentions = vectorizer.transform(mentions_texts.tolist())
        # binarize
        X_mentions[X_mentions>0] = 1
        # apply z-score values to each row in `X_mentions` as weights
        X_mentions_scores = X_mentions @ fw_vals[:, np.newaxis]
        # normalize for mention length
        X_mentions_scores /= X_mentions.sum(axis=1)

        focal_mentions['score'] = X_mentions_scores[:, 0]
        # rank = focal_mentions['score'].argsort()[::-1]

        examples = focal_mentions[focal_mentions['score']>1.96].copy()
        examples.loc[:, 'text_norm'] = examples['text'].apply(lambda x: ' '.join(analyzer(x)).strip())
        examples = examples.drop_duplicates('text_norm').sort_values('score', ascending=False).head(400)

        attribute_cats_examples[lab] = examples

    with open(attribute_cats_fws_fp, 'wb') as f:
        pickle.dump(attribute_cats_fws, f)
    with open(attribute_cats_examples_fp, 'wb') as f:
        pickle.dump(attribute_cats_examples, f)

    # evaluate how well the top-k terms for each attribute category predict the presence of that attribute in mentions
    # NOTE: The idea here is to address the question how well a data-driven dictionary would solve the same task

    df_universal.loc[:, 'tokens'] = df_universal['text'].apply(lambda x: analyzer(x))
    keywords_performances = {}
    for a, fws in attribute_cats_fws.items():
        keywords_performances[a] = {}
        for c in range(10, 51, 10):
            terms = attribute_cats_fws[a].head(c).word.tolist()
            keywords_performances[a][c] = df_universal.groupby(a)["tokens"].apply(lambda grp: np.mean([any(t in terms for t in toks if t ) for toks in grp])).to_dict()

    keywords_performances_df = pd.concat({
        a: pd.DataFrame(d)
        for a, d in keywords_performances.items()
    }).reset_index(names=['attribute_category', 'label'])

    keywords_performances_df.to_pickle(attribute_fw_topterm_performances_fp)

else:
    with open(attribute_cats_fws_fp, 'rb') as f:
        attribute_cats_fws = pickle.load(f)
    with open(attribute_cats_examples_fp, 'rb') as f:
        attribute_cats_examples = pickle.load(f)
    keywords_performances_df = pd.read_pickle(attribute_fw_topterm_performances_fp)




# TABLE C3
tbl_cap = r"Most distinctive terms for each attribute category, based on the ten tokens with the highest fighting words $z$-scores. The $z$-scores, reported in parentheses, indicate how strongly a term is associated with mentions in that attribute category compared to social group mentions that do feature attributes from this category. \emph{Note:} See \ref{figureC02} and \ref{figureC03} for visual presentations of the data."

tab = {}
for a, subdf in attribute_cats_fws.items():
    k = a.split('__')[0], attribute_category_names_map[a]
    words = subdf.head(10).apply(lambda row: f"{row['word']}~({row['score']:0.1f})", axis=1).tolist()
    tab[k] = pd.DataFrame({'top-10 most distinctiveterms': ', '.join(words)}, index=[0])

tab = pd.concat(tab)

tab = tab.reset_index(level=[0, 1], names=['dimension', 'category'])
tab.loc[tab['dimension']=='non-economic', 'dimension'] = 'non-economic'
tab = tab.set_index(['dimension', 'category'])
tab.index.names = [None, None]

tab = latex_table(tab, caption=tbl_cap, index=True, column_format='l L{4.5cm} L{0.85\\textwidth}', escape=False, label='tableC03')
save_table(tab, 'tableC03.tex', dest=tables_path)




# FIGURE C2
fig_cap = r"Most distinctive words of social group mentions categorized as featuring a specific economic attribute. Values plotted are $z$-scores from ``fighting words'' analysis computed for mentions in focal category vs. all other non-universal mentions. Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

fig, axes = plt.subplots(1, len(econ_attrs), figsize=(len(econ_attrs)*3, 4), sharey=False)

for i, attr, in enumerate(econ_attrs):
    ax = axes[i]
    subdf = attribute_cats_fws[attr]
    top_positive = subdf.nlargest(20, 'score').sort_values('score', ascending=True)
    ttl = attribute_category_names_map[attr]
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.barh(range(len(top_positive)), top_positive['score'], color='gray', zorder=2)
    ax.set_yticks(range(len(top_positive)))
    ax.set_yticklabels(top_positive['word'])
    ax.set_xlabel('z-score', fontsize=11)
    ax.set_title(ttl, fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')
    # plt.setp(ax.get_yticklabels(), ha='left')
    ax.set_xlim(0, 140)
    # ax.invert_xaxis()  # Invert to show positive values extending left

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureC02.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE C3
fig_cap = "Most distinctive words for social group mentions categorized as featuring a specific non-economic attribute. Values plotted are $z$-scores from ``fighting words'' analysis computed for mentions in focal category vs. all other non-universal mentions. Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

n_cols = len(nonecon_attrs) // 2
n_rows = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*4), sharey=False, sharex=False)
axes = axes.flatten()

for i, attr, in enumerate(nonecon_attrs):
    ax = axes[i]
    subdf = attribute_cats_fws[attr]
    top_positive = subdf.nlargest(20, 'score').sort_values('score', ascending=True)
    ttl = attribute_category_names_map[attr]
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.barh(range(len(top_positive)), top_positive['score'], color='gray', zorder=2)
    ax.set_yticks(range(len(top_positive)))
    ax.set_yticklabels(top_positive['word'])
    ax.set_title(ttl, fontweight='bold', fontsize=12)
    if i % n_cols == 0:
        ax.set_ylabel('distinctive words', fontsize=10)
    if i >= n_cols * (n_rows - 1):
        ax.set_xlabel('z-score', fontsize=11)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position('right')
    # plt.setp(ax.get_yticklabels(), ha='left')
    ax.set_xlim(0, 140)
    # ax.invert_xaxis()  # Invert to show positive values extending left

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureC03.pdf', caption=fig_cap, dest=figures_path) 




# TABLE C4
tbl_cap = r"Performance of the top $k$ terms for each attribute category in predicting the presence of that attribute in mentions. The table reports the proportion of mentions in each attribute category that contain at least one of the top $k$ terms for that category. The top $k$ terms are determined based on the ten tokens with the highest fighting words $z$-scores."

tab = keywords_performances_df.query("label==1").copy()
tab['dimension'] = tab['attribute_category'].map(lambda x: x.split('__')[0])
tab.loc[tab['dimension']=='non-economic', 'dimension'] = 'non-economic'
tab = tab[['dimension', 'attribute_category', *keywords_performances_df.columns[2:].tolist()]]
tab = tab.sort_values(['dimension', 10], ascending=[True, False])
tab.attribute_category = tab.attribute_category.map(attribute_category_names_map)
tab = tab.set_index(['dimension', 'attribute_category'])
tab.index.names = [None, None]
tab.columns = pd.MultiIndex.from_tuples([(f"top $k$ terms", f"{c}") for c in tab.columns], names=['', ''])

tab = format_table(tab, caption=tbl_cap, index=True, label='tableC04')
save_table(tab, 'tableC04.tex', dest=tables_path) 




# fp = annotations_path / "exdata" / 'thau2019_appeals_appeal.csv'


# if not fp.exists():
#     # Inference on Mads Thau's annotations
#     df_thau_raw = pd.read_csv("https://github.com/haukelicht/group_mention_detection/raw/refs/heads/main/replication/data/exdata/thau2019/thau2019_appeals_appeal.csv", encoding='latin1')
#     df_thau = df_thau_raw[['objid', 'objtype', 'objdim']].drop_duplicates()
#     df_thau.columns = ['mention', 'group_type', 'group_category']

#     # NOTE: Thau's coders used (..) to separate nested mentions
#     idxs = df_thau.mention.str.contains('(', regex=False)
#     # print(df_thau[idxs])
#     df_thau.loc[idxs, 'mention'] = df_thau.loc[idxs, 'mention'].str.replace('(', '').str.replace(')', '')

#     # NOTE: Thau's coders used .... to fix interrupted mentions
#     idxs = df_thau.mention.str.contains('...', regex=False)
#     # print(df_thau[idxs])
#     df_thau.loc[idxs, 'mention'] = df_thau.loc[idxs, 'mention'].str.replace('...', ' ')
#     df_thau.loc[idxs, 'mention'] = df_thau.loc[idxs, 'mention'].str.replace('…', ' ')

#     # NOTE: Thau's coders used [...] to resolve coreferences
#     idxs = df_thau.mention.str.contains('[', regex=False)
#     # print(df_thau[idxs])
#     df_thau.loc[idxs, 'mention'] = df_thau.loc[idxs, 'mention'].str.replace(r'\[[^\]]+?\]', ' ', regex=True)
#     df_thau['mention'] = df_thau.mention.str.replace(r'\s+', ' ', regex=True)

#     # create dir if needed
#     fp.parent.mkdir(parents=True, exist_ok=True)
#     df_thau.reset_index(drop=True, inplace=True)
#     df_thau.to_csv(fp, index=False)
# else:
#     df_thau = pd.read_csv(fp)

def fix_mojibake(s):
    # undo UTF-8 text that was mistakenly decoded as Latin-1 and re-saved as UTF-8 (e.g. "banks\xe2\x80\x99" -> "banks'")
    if not isinstance(s, str):
        return s
    try:
        return s.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s

fp = data_path / "exdata" / 'thau2019_group_appeals_annotation.csv'
df_thau = pd.read_csv(fp)
df_thau['mention'] = df_thau['mention'].apply(fix_mojibake)




fp = intermediate_path / 'thau2019_attribute_classifications.pkl'

if not fp.exists():
    import torch
    from setfit import SetFitModel

    # apply our classifiers
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model_name = "haukelicht/all-mpnet-base-v2_economic-attributes-classifier"
    econ_classifier = SetFitModel.from_pretrained(model_name, force_download=True)
    # TODO: add revision number to ensure reproducibility
    econ_classifier.to(device);

    model_name = "haukelicht/all-mpnet-base-v2_noneconomic-attributes-classifier"
    nonecon_classifier = SetFitModel.from_pretrained(model_name, force_download=True)
    # TODO: add revision number to ensure reproducibility
    nonecon_classifier.to(device);

    label_cols = econ_classifier.labels + nonecon_classifier.labels

    def predict_attributes(x: list[str]) -> pd.DataFrame:
        with torch.no_grad():
            econ_predictions = econ_classifier.predict(x, as_numpy=True)
            nonecon_predictions = nonecon_classifier.predict(x, as_numpy=True)
        preds = np.concatenate([econ_predictions, nonecon_predictions], axis=1)
        return pd.DataFrame(preds, columns=label_cols)

    preds = predict_attributes(df_thau.mention.to_list())
    df_thau_preds = pd.concat([df_thau.reset_index(drop=True), preds], axis=1)

    fp.parent.mkdir(parents=True, exist_ok=True)
    df_thau_preds.to_pickle(fp)

    del nonecon_classifier
    del econ_classifier

    gc.collect()
else:
    label_cols = list(attribute_category_names_map.keys())
    df_thau_preds = pd.read_pickle(fp)




df_thau_preds['mention'] = df_thau_preds['mention'].apply(fix_mojibake)




# compute conditional probability: Pr( attribute present | Thau's category )
these_thau_cats = ["Professional group", "Social group"]
cpr = df_thau_preds.query("group_type in @these_thau_cats")
cpr.loc[cpr.group_type=='Professional group', 'group_category'] = 'Professional group'
cpr = cpr.groupby('group_category')[label_cols].mean()
cpr[cpr<=0.009] = np.nan




thau_group_cats = cpr.index.sort_values().unique().tolist()
thau_group_cats.remove('Other')
thau_group_cats.append('Other')




# FIGURE C4
fig_cap = r"Correspondence of Thau's social group categorizations with our group attribute classifications. Numbers report the probability that a mention assigned by Thau's coders to one of his social group categories (shown on y-axis) has been labeled as featuring a given social attribute in our classification scheme (shown on x-axis). \emph{Note:} Values below 0.009 not plotted to ease readability."

fig, axes = plot_heatmap(
    cpr.rename(columns=attribute_category_names_map), 
    panel_groups=[None, (econ_attr_names, nonecon_attr_names)],
    cmap='RdPu',
    clims=(0, 1.0),
    clegend_title='conditional probability\nPr(our classification | Thau\'s classification)',
    figsize_multiplier = (0.44, 0.46),
)
# reduce y-axis label size to fit better
for ax in axes:
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
axes[0].set_ylabel('Thau\'s group categories', fontsize=10, labelpad=10, fontweight='bold')
# plt.show()

save_figure(fig, 'figureC04.pdf', caption=fig_cap, dest=figures_path) 




# filter verbatim mentions that were assigned into multible group type categories
thau_ambigous_labels_examples = df_thau.query('group_type == "Social group"').groupby(['mention', 'group_type']).filter(lambda x: x.group_category.nunique()>1).groupby(['mention', 'group_type'])['group_category'].agg(set).sort_index().reset_index(name='categories')
thau_ambigous_labels_examples['categories_str'] = thau_ambigous_labels_examples['categories'].apply(lambda cats: ' + '.join(sorted(cats)))




# print('N =', (df_thau.group_type=="Social group").sum())
# print('N =', len(thau_ambigous_labels_examples), f'; share = {len(thau_ambigous_labels_examples)/(df_thau.group_type == "Social group").sum():0.3f}')




# TABLE C5
tbl_cap = r"Examples of verbatim group mentions with conflicting group category annotations in data by Thau (2019) for category combinations that were confused more than 10 times. Each row shows (at most) three examples of verbatim group mentions that have been categorized into one of the following categories in different sentence contexts in the corpus and thus have \"conflicting\" labels. $N$ reports the total number of mentions that exhibit the shown category confusion (in all 3,571 mentions)."

expls_df = thau_ambigous_labels_examples.copy()
expls_df['n_attrs'] = expls_df['categories'].apply(len)
expls_df[['cat1', 'cat2']] = expls_df['categories'].apply(lambda cats: pd.Series(sorted(list(cats))[:2]))
expls_df = expls_df.groupby(['categories_str']).apply(lambda x: x.assign(n=len(x)).sample(min(len(x), 3), random_state=42)).reset_index(drop=True)
expls_df = expls_df.groupby(['categories_str', 'cat1', 'cat2', 'n_attrs', "n"]).agg({'mention': lambda x : '; '.join(f"``{m}''"for m in x)}).reset_index()
expls_df.sort_values(['n_attrs', "n", 'categories_str'], ascending=[True, False, True])
expls_df.drop_duplicates(['cat1', 'cat2']).query('cat1!="Other" and cat2!="Other"').reset_index(drop=True)
tab = expls_df.query('n> 10').sort_values(['cat1', 'cat2'])
tab = tab[['categories_str', 'n', 'mention']]
tab.columns = ["attribute combination", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC05')
save_table(tab, 'tableC05.tex', dest=tables_path) 


#### mentions in Thau's Economic class category



thau_econ_class_mentions = df_thau_preds.query("group_category=='Economic class'").copy()
thau_econ_class_mentions['labels'] = thau_econ_class_mentions.apply(lambda row: ' + '.join([l for c, l in attribute_category_names_map.items() if row[c]==1]), axis=1)
tmp = thau_econ_class_mentions.\
    groupby('labels').\
    agg({'group_type': 'count', 'mention': lambda x: list(x.sample(min(len(x), 5), random_state=1 ))}).\
    reset_index().\
    rename(columns={'group_type': 'count'}).\
    sort_values('count', ascending=False)
tmp = tmp[tmp['labels']!='']
tmp['n_attributes'] = tmp['labels'].str.count(' \+ ')+1
tmp['combinational'] = tmp['n_attributes'] > 1
tmp['examples'] = tmp['mention'].apply(lambda ms: '; '.join([f"``{m}''" for m in ms]))




cnts = tmp.groupby('combinational').agg({'count': 'sum'})
props = cnts / cnts.sum()
# print(props)




# TABLE C6
tbl_cap = r"Social group mentions in Thau's \emph{Economic class} group category that are assigned to a single attribute category according to our scheme and their absolute frequency. \emph{Note:} Table only reports results for the six most prevalent attribute categories."

tab = tmp.query("n_attributes == 1").head(6)[['labels', 'count', 'examples']]
tab.columns = ["attribute category", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC06')
save_table(tab, 'tableC06.tex', dest=tables_path) 




# TABLE C7
tbl_cap = r"Social group mentions in Thau's \emph{Economic class} that are assigned to two or more attribute categories according to our scheme and their absolute frequency. \emph{Note:} Table only reports results for the six most prevalent attribute combinations."

tab = tmp.query("n_attributes >= 2").head(6)[['labels', 'count', 'examples']]
tab.columns = ["attribute combination", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC07')
save_table(tab, 'tableC07.tex', dest=tables_path) 


#### mentions in Thau's Family category



thau_gender_mentions = df_thau_preds.query("group_category=='Gender'").copy()
thau_gender_mentions['labels'] = thau_gender_mentions.apply(lambda row: ' + '.join([l for c, l in attribute_category_names_map.items() if row[c]==1]), axis=1)
tmp = thau_gender_mentions.\
    groupby('labels').\
    agg({'group_type': 'count', 'mention': lambda x: list(x.sample(min(len(x), 5), random_state=1 ))}).\
    reset_index().\
    rename(columns={'group_type': 'count'}).\
    sort_values('count', ascending=False)
tmp = tmp[tmp['labels']!='']
tmp['n_attributes'] = tmp['labels'].str.count(' \+ ')+1
tmp['combinational'] = tmp['n_attributes'] > 1
tmp['examples'] = tmp['mention'].apply(lambda ms: '; '.join([f"``{m}''" for m in ms]))




cnts = tmp.groupby('combinational')['count'].sum()
props = cnts / cnts.sum()
# print(props)




# TABLE C8
tbl_cap = r"Social group mentions in Thau's \emph{Gender} group category that are assigned to only one attribute category according to our scheme and their absolute frequency. \emph{Note:} Table only reports the results for the six most prevalent attribute categories."

tab = tmp.query("n_attributes == 1").head(6)[['labels', 'count', 'examples']]
tab.columns = ["attribute", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC08')
save_table(tab, 'tableC08.tex', dest=tables_path) 




# TABLE C9
tbl_cap = r"Social group mentions in Thau's \emph{Gender} group category that are assigned to two attribute categories according to our scheme and their absolute frequency. \emph{Note:} Table only reports the results for the six most prevalent attribute combinations."

tab = tmp.query("n_attributes == 2").head(6)[['labels', 'count', 'examples']]
tab.columns = ["attribute combination", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC09')
save_table(tab, 'tableC09.tex', dest=tables_path) 




# print((df_thau_preds.query("group_type=='Social group'").group_category=='Other').mean())




thau_other_mentions = df_thau_preds.query("group_category=='Other'").copy()
thau_other_mentions['labels'] = thau_other_mentions.apply(lambda row: ' + '.join([l for c, l in attribute_category_names_map.items() if row[c]==1]), axis=1)
tmp = thau_other_mentions.\
    groupby('labels').\
    agg({'group_type': 'count', 'mention': lambda x: list(x.sample(min(len(x), 5), random_state=1 ))}).\
    reset_index().\
    rename(columns={'group_type': 'count'}).\
    sort_values('count', ascending=False)
tmp['n_attributes'] = tmp['labels'].str.count(' \+ ')+1
tmp['any_attributes'] = tmp['labels']!=''
tmp.loc[~tmp['any_attributes'], 'n_attributes'] = 0
tmp['combinational'] = tmp['n_attributes'] > 1
tmp['examples'] = tmp['mention'].apply(lambda ms: '; '.join([f"``{m}''" for m in ms]))




cnts = tmp.groupby(['any_attributes', 'combinational'])['count'].sum()
props = cnts / cnts.sum()
# print(props)




cnts = tmp.query('any_attributes').groupby('combinational')['count'].sum()
props = cnts / cnts.sum()
# print(props)




# TABLE C10
tbl_cap = r"Social group mentions in Thau's \emph{Gender} group category that are assigned to two attribute categories according to our scheme and their absolute frequency. \emph{Note:} Table only reports the results for the six most prevalent attribute combinations."

tab = tmp.query("n_attributes >= 1").head(6)[['labels', 'count', 'examples']]
tab.columns = ["attribute(s)", "$N$", "examples"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{2.2in} l L{3in}', label='tableC10')
save_table(tab, 'tableC10.tex', dest=tables_path) 


### Comparison to Horne et al.'s group category annotations



fps = [
    intermediate_path / 'horne_predictions_sample.pkl',
    intermediate_path / 'horne_predictions_econ_occ-prof_attribute_sample.pkl',
    intermediate_path / 'horne_predictions_econ_employment_attribute_sample.pkl',
    intermediate_path / 'horne_predictions_nonecon_gender-sexuality_attribute_sample.pkl'
]

UPDATE = False
need_to_predict = UPDATE or not all(fp.exists() for fp in fps)

if need_to_predict:
    from transformers import pipeline, AutoConfig

    # Use a pipeline as a high-level helper
    fp = data_path / 'labeled_mentions_with_party_metadata.feather'
    df_labeled = pd.read_feather(fp)

    # NOTE: subset to contries with languages in which Horne et al. (2022) ahve evaluated their model
    fp = manifestos_path / "all_manifesto_sentences.feather"
    sentences_df = pd.read_feather(fp)[['sentence_id', 'lang']]
    df_labeled = df_labeled.merge(sentences_df, how='left', on='sentence_id')

    focal_langs = {
        "en": "English",
        "de": "German",
        "da": "Danish",
        "es": "Spanish",
        "nl": "Dutch",
        "fr": "French",
        "it": "Italian",
        "sv": "Swedish"
    }
    df_labeled = df_labeled[df_labeled.lang.isin(focal_langs.keys())]

    # get sample and apply to all
    df_sample = df_labeled.sample(frac=0.1, random_state=42).reset_index(drop=True)

    model_id = "rwillh11/mdeberta_groups_2.0"
    config = AutoConfig.from_pretrained(model_id)

    # print(config.problem_type)
    pipe = pipeline("text-classification", model=model_id, top_k=None, function_to_apply="sigmoid")

    label_cols_horne = list(config.label2id.keys())

    def predict_horne(x: list[str]) -> pd.DataFrame:
        preds = pipe(x, batch_size=16)
        return pd.concat([pd.DataFrame({lab['label']: lab['score']>=0.5 for lab in e}, index=[i]) for i, e in enumerate(preds)])[label_cols_horne].astype(int)

    df_sample[label_cols_horne] = predict_horne(df_sample.text.to_list())
    df_sample.to_pickle(fps[0])

    cols = ["country_iso3c", "party_id", "party_family", "date", "sentence_id", "sentence_text", "mention_id", "text"]

    # apply to occupation/profession instances
    horne_predictions_econ_occprof_attribute_sample = df_labeled.query('economic__occupation_profession==1')[cols].copy().reset_index(drop=True)
    # horne_predictions_econ_occprof_attribute_sample = horne_predictions_econ_occprof_attribute_sample.sample(5_000, random_state=42).reset_index(drop=True)
    horne_predictions_econ_occprof_attribute_sample[label_cols_horne] = predict_horne(horne_predictions_econ_occprof_attribute_sample.text.to_list())
    horne_predictions_econ_occprof_attribute_sample.to_pickle(fps[1])

    # apply to employment status instances
    horne_predictions_econ_employment_attribute_sample = df_labeled.query('economic__employment_status==1')[cols].copy()
    # horne_predictions_econ_employment_attribute_sample = horne_predictions_econ_employment_attribute_sample.sample(5_000, random_state=42).reset_index(drop=True)
    horne_predictions_econ_employment_attribute_sample[label_cols_horne] = predict_horne(horne_predictions_econ_employment_attribute_sample.text.to_list())
    horne_predictions_econ_employment_attribute_sample.to_pickle(fps[2])

    # apply to gender/sexuality instances
    horne_predictions_nonecon_gender_attribute_sample = df_labeled.query('noneconomic__gender_sexuality==1')[cols].copy().reset_index(drop=True)
    # horne_predictions_nonecon_gender_attribute_sample = horne_predictions_nonecon_gender_attribute_sample.sample(5_000, random_state=42).reset_index(drop=True)
    horne_predictions_nonecon_gender_attribute_sample[label_cols_horne] = predict_horne(horne_predictions_nonecon_gender_attribute_sample.text.to_list())
    horne_predictions_nonecon_gender_attribute_sample.to_pickle(fps[3])
else:
    df_sample = pd.read_pickle(fps[0])
    horne_predictions_econ_occprof_attribute_sample = pd.read_pickle(fps[1])
    horne_predictions_econ_employment_attribute_sample = pd.read_pickle(fps[2])
    horne_predictions_nonecon_gender_attribute_sample = pd.read_pickle(fps[3])

    label_cols_horne = [c for c in df_sample.columns if re.match(r'^[A-Z]', c)]




# TODO: see if this is repeatedly used elsewhere
from src.reporting import label_assoc_matrices




assoc = label_assoc_matrices(df_sample, label_cols, label_cols_horne)

npmi = assoc["npmi"]          # nPMI in [-1, 1]
ppmi = assoc["ppmi"]          # PPMI (>=0)
counts = assoc["cooc_counts"] # co-occurrence counts
p_b_given_a = assoc["p_b_given_a"]




# compute a chi2 test on the co-occurrence counts matrix
chi2, p, dof, ex = chi2_contingency(counts)

# compute cramer's V
cramers_v = np.sqrt(chi2 / (assoc['marginals']['N'] * (min(len(label_cols), len(label_cols_horne)) - 1)))

# print(f"Chi2 test on co-occurrence counts matrix: chi2={chi2:0.2f}, p={p:0.3e}, dof={dof}, Cramér's V={cramers_v:0.3f}")




# FIGURE C5
fig_cap = r"Patterns of convergence between group mention classifications of our attribute-centered classifier and Horne et al.'s group category classifier. The figure reports normalized \emph{Pointwise Mutual Information} (nPMI) values that measure the strength of association of label classes on a scale ranging from -1 (systematic disassociation) through 0 (independence) to +1 (perfect co-occurrence).  y-axis labels indicate the attribute category in our scheme; x-axis labels the group categories in Horne et al.'s scheme. Plot panels separated by economic and non-economic attributes. \emph{Note:} heatmap columns sorted (for each attribute dimension) using hierarchical clustering to better reveal conceptual overlap."

# Cluster economic attributes
pdat_econ = npmi.loc[econ_attrs].rename(index=attribute_category_names_map)
cl = sns.clustermap(pdat_econ)
plt.close()
pdat_econ = pdat_econ.iloc[cl.dendrogram_row.reordered_ind, cl.dendrogram_col.reordered_ind]

# Cluster economic attributes
pdat_nonecon = npmi.loc[nonecon_attrs].rename(index=attribute_category_names_map)
cl = sns.clustermap(pdat_nonecon)
plt.close()
pdat_nonecon = pdat_nonecon.iloc[cl.dendrogram_row.reordered_ind, cl.dendrogram_col.reordered_ind]

# Create combined figure with gridspec
heights = [len(econ_attrs), len(nonecon_attrs)+3.3]
fig = plt.figure(figsize=(12, (len(econ_attrs) + len(nonecon_attrs))*0.5))
gs = fig.add_gridspec(2, 1, height_ratios=heights, hspace=1.2)

# Create axes for the two heatmaps
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# Plot economic attributes
sns.heatmap(pdat_econ, 
            fmt='.2f',
            square=True,
            cmap='PiYG',
            vmin=-1, vmax=1,
            cbar=False,
            linewidths=1,
            linecolor='white',
            ax=ax1,
            xticklabels=True,
            yticklabels=True)

# Move x-axis to top and rotate labels
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='left')
ax1.set_xlabel(None)
ax1.set_ylabel(None)

# Plot non-economic attributes
sns.heatmap(pdat_nonecon, 
            fmt='.2f',
            square=True,
            cmap='PiYG',
            vmin=-1, vmax=1,
            cbar=True,
            cbar_kws={'orientation': 'horizontal', 'shrink': 0.6, 'pad': 0.1},
            linewidths=1,
            linecolor='white',
            ax=ax2,
            xticklabels=True,
            yticklabels=True
            )
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position('top')
plt.setp(ax2.get_xticklabels(), rotation=45, ha='left')
# plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_xlabel(None)
ax2.set_ylabel(None)

# set colorbar label
cbar = ax2.collections[0].colorbar
cbar.set_label('normalized Pointwise Mutual Information (nPMI)', rotation=0, labelpad=6, ha='center')

# plt.show()
save_figure(fig, 'figureC05.pdf', caption=fig_cap, dest=figures_path) 


#### _Occupation/profession_-related social group references



occupation_cats = [
	'Caregivers',
	'Civil Servants',
	'Education Professionals',
	'Employees And Workers',
	'Employers And Business Owners',
	'Farmers',
	'Health Professionals',
	'Investors And Stakeholders',
	'Law Enforcement Personnel',
	'Manual And Service Workers',
	'Military Personnel',
	'Politicians',
	'Sociocultural Professionals',
	'White Collar Workers',
]

tmp = horne_predictions_econ_occprof_attribute_sample.copy()




# print(tmp[occupation_cats].any(axis=1).mean())




vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=stopwords,
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)

idxs = horne_predictions_econ_occprof_attribute_sample[occupation_cats].any(axis=1)
fw = compute_fighting_words(
    l1=horne_predictions_econ_occprof_attribute_sample.loc[ idxs, 'text'].to_list(),
    l2=horne_predictions_econ_occprof_attribute_sample.loc[~idxs, 'text'].to_list(),
    cv=vectorizer,
)
fw_ours_vs_horne_occupation = pd.DataFrame(fw, columns=['word', 'score']).sort_values('score', ascending=False)




# FIGURE C6
fig_cap = r"Most distinctive words for mentions labeled as featuring occupation/profession as an attribute by our classifier depending on whether Horne et al.'s classifier has classified them into (at least) one of their occupation/profession-related categories (left) or not (right). Values plotted are $z$-scores from ``fighting words'' analysis of occupation/profession mentions. Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

# Get top 20 lowest (most negative) and highest (most positive) scores
top_negative = fw_ours_vs_horne_occupation.nsmallest(20, 'score').sort_values('score', ascending=False)
top_positive = fw_ours_vs_horne_occupation.nlargest(20, 'score').sort_values('score', ascending=True)

# Create two-column layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

# Left plot: positive scores (distinctive for classified by Horne et al.)
ttl = r"categorized by both classifiers"
ax1.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax1.barh(range(len(top_positive)), top_positive['score'], color='#1b9e77', zorder=2)
ax1.set_yticks(range(len(top_positive)))
ax1.set_yticklabels(top_positive['word'])
ax1.set_xlabel('z-score', fontsize=11)
ax1.set_title(ttl, fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
plt.setp(ax1.get_yticklabels(), ha='left')
ax1.set_xlim(0, 25)
ax1.invert_xaxis()  # Invert to show negative values extending left

# Right plot: negative scores (distinctive for NOT classified by Horne et al.)
ttl = r"categorized only by our classifier"
ax2.axvline(x=-1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax2.barh(range(len(top_negative)), top_negative['score'], color='#d95f02', zorder=2)
ax2.set_yticks(range(len(top_negative)))
ax2.set_yticklabels(top_negative['word'])
ax2.set_xlabel('z-score', fontsize=11)
ax2.set_title(ttl, fontweight='bold', fontsize=12)
ax2.set_xlim(-25, 0)
ax2.invert_xaxis()  # Invert to show negative values extending left

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureC06.pdf', caption=fig_cap, dest=figures_path) 




# TABLE C11
tbl_cap = r"Examples of social group mentions labeled as featuring occupation/profession as an attribute by our classifier that were not assigned to any of Horne et al.'s occupation-related group categories by their classifier. Values computed by summing ``fighting words'' scores as weights of mentions' tokens, normalized by number of tokens."
fw_lookup = {r['word']: r['score'] for r in fw_ours_vs_horne_occupation.to_dict(orient='records')}
fw_vals = np.array([fw_lookup[f] for f in vectorizer.get_feature_names_out()])
analyzer = vectorizer.build_analyzer()

# vectorize mentions 
mentions = horne_predictions_econ_occprof_attribute_sample[~idxs].reset_index(drop=True)
mentions_texts = mentions['text']
X_mentions = vectorizer.transform(mentions_texts.tolist())
# binarize
X_mentions[X_mentions>0] = 1
# apply z-score values to each row in `X_mentions` as weights
X_mentions_scores = X_mentions @ fw_vals[:, np.newaxis]
# normalize for mention length
X_mentions_scores /= X_mentions.sum(axis=1)

mention_scores = X_mentions_scores[:, 0]
rank = mention_scores.argsort()#[::-1]

n_ = 20
tab = horne_predictions_econ_occprof_attribute_sample.loc[~idxs, ['text', *label_cols_horne]].iloc[rank]
tab['score'] = mention_scores[rank]
tab = tab[tab['score']<-1.96]
tab['text_norm'] = tab['text'].apply(lambda x: ' '.join(analyzer(x)).strip())
tab = tab.drop_duplicates('text_norm')# .head(n_).reset_index(drop=True)
tab = tab.sample(n_, weights=tab['score'].abs(), random_state=42)
tab.loc[:, 'horne'] = tab.iloc[:, 1:].apply(lambda row: '; '.join([l for l in label_cols_horne if row[l]==1]), axis=1)
tab = tab[['text', 'score', 'horne']].sort_values('score')
tab.columns = ["Mention", "$z$-score", "Horne et al. classification"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{3in} l l', label='tableC11')
save_table(tab, 'tableC11.tex', dest=tables_path) 


#### _Gender/sexuality_-related social group references



horne_gender_sexuality_cats = [
	'Lgbtqi',
	'Men',
	'Women',
]

tmp = horne_predictions_nonecon_gender_attribute_sample.copy()




# print(tmp[horne_gender_sexuality_cats].any(axis=1).mean())




vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=stopwords,
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)

idxs = horne_predictions_nonecon_gender_attribute_sample[horne_gender_sexuality_cats].any(axis=1)
fw = compute_fighting_words(
    l1=horne_predictions_nonecon_gender_attribute_sample.loc[ idxs, 'text'].to_list(),
    l2=horne_predictions_nonecon_gender_attribute_sample.loc[~idxs, 'text'].to_list(),
    cv=vectorizer,
)
fw_ours_vs_horne_gender = pd.DataFrame(fw, columns=['word', 'score']).sort_values('score', ascending=False)




# FIGURE C7
fig_cap = r"Top 20 most distinctive words for gender/sexuality mentions classified by Horne et al. (right) vs. not classified (left)."

# Get top 20 lowest (most negative) and highest (most positive) scores
top_negative = fw_ours_vs_horne_gender.nsmallest(20, 'score').sort_values('score', ascending=False)
top_positive = fw_ours_vs_horne_gender.nlargest(20, 'score').sort_values('score', ascending=True)

# Create two-column layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

# Left plot: positive scores (distinctive for classified by Horne et al.)
ttl = r"categorized by both classifiers"
ax1.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax1.barh(range(len(top_positive)), top_positive['score'], color='#1b9e77', zorder=2)
ax1.set_yticks(range(len(top_positive)))
ax1.set_yticklabels(top_positive['word'])
ax1.set_xlabel('Score (z-score)', fontsize=11)
ax1.set_title(ttl, fontweight='bold', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')
plt.setp(ax1.get_yticklabels(), ha='left')
ax1.set_xlim(0, 35)
ax1.invert_xaxis()  # Invert to show negative values extending left

# Right plot: negative scores (distinctive for NOT classified by Horne et al.)
ttl = r"categorized only by our classifier"
ax2.axvline(x=-1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
ax2.barh(range(len(top_negative)), top_negative['score'], color='#d95f02', zorder=2)
ax2.set_yticks(range(len(top_negative)))
ax2.set_yticklabels(top_negative['word'])
ax2.set_xlabel('Score (z-score)', fontsize=11)
ax2.set_title(ttl, fontweight='bold', fontsize=12)
ax2.set_xlim(-35, 0)
ax2.invert_xaxis()  # Invert to show negative values extending left

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureC07.pdf', caption=fig_cap, dest=figures_path) 




# TABLE C12
tbl_cap = r"Examples of social group mentions labeled as featuring gender/sexuality as an attribute by our classifier that were not assigned to any of Horne et al.'s Men, Women, or LGBTQI categories by their classifier. Values computed by summing ``fighting words'' scores as weights of mentions' tokens, normalized by number of tokens."

fw_lookup = {r['word']: r['score'] for r in fw_ours_vs_horne_gender.to_dict(orient='records')}
fw_vals = np.array([fw_lookup[f] for f in vectorizer.get_feature_names_out()])
analyzer = vectorizer.build_analyzer()

# vectorize mentions 
mentions = horne_predictions_nonecon_gender_attribute_sample[~idxs].reset_index(drop=True)
mentions_texts = mentions['text']
X_mentions = vectorizer.transform(mentions_texts.tolist())
# binarize
X_mentions[X_mentions>0] = 1
# apply z-score values to each row in `X_mentions` as weights
X_mentions_scores = X_mentions @ fw_vals[:, np.newaxis]
# normalize for mention length
X_mentions_scores /= X_mentions.sum(axis=1)

mention_scores = X_mentions_scores[:, 0]
rank = mention_scores.argsort()

n_ = 20
tab = horne_predictions_nonecon_gender_attribute_sample.loc[~idxs, ['text', *label_cols_horne]].iloc[rank]
tab['score'] = mention_scores[rank]
tab = tab[tab['score']<-1.64]
tab['text_norm'] = tab['text'].apply(lambda x: ' '.join(analyzer(x)).strip())
tab = tab.drop_duplicates('text_norm')#.head(n_).reset_index(drop=True)
tab = tab.sample(n_, weights=tab['score'].abs(), random_state=1)
tab.loc[:, 'horne'] = tab.iloc[:, 1:].apply(lambda row: '; '.join([l for l in label_cols_horne if row[l]==1]), axis=1)
tab[['text', 'score', 'horne']].sort_values('score')
tab = tab[['text', 'score', 'horne']].sort_values('score')
tab.columns = ["Mention", "$z$-score", "Horne et al. classification"]

tab = latex_table(tab, caption=tbl_cap, column_format='p{3in} l l', label='tableC12')
save_table(tab, 'tableC12.tex', dest=tables_path) 


# Main analyses 



df.loc[:, 'is_universal_'] = ~df[econ_attrs+nonecon_attrs].any(axis=1)
universal_vs_others = df.value_counts('is_universal_', normalize=True)
universal_vs_others = universal_vs_others.to_frame()
universal_vs_others.loc[:, 'count'] = df.loc[:, 'is_universal_'].value_counts()
universal_vs_others.index = universal_vs_others.index.map({True: 'universal', False: 'has attributes'})

tmp = df[~df['is_universal_']].copy()
tmp.loc[:, 'single_attr_'] = tmp[econ_attrs+nonecon_attrs].sum(axis=1) == 1
single_vs_multi_attrs = tmp.loc[:, 'single_attr_'].value_counts(normalize=True)
single_vs_multi_attrs = single_vs_multi_attrs.to_frame()
single_vs_multi_attrs.loc[:, 'count'] = tmp.loc[:, 'single_attr_'].value_counts()
single_vs_multi_attrs.index = single_vs_multi_attrs.index.map({True: 'single-attribute', False: 'multi-attribute'})

tmp1 = tmp[tmp.loc[:, 'single_attr_']].copy()
tmp1.loc[:, 'econ_attr_'] = tmp1[econ_attrs].sum(axis=1) == 1
single_attr_econ_vs_nonecon = tmp1.loc[:, 'econ_attr_'].value_counts(normalize=True)
single_attr_econ_vs_nonecon = single_attr_econ_vs_nonecon.to_frame()
single_attr_econ_vs_nonecon.loc[:, 'count'] = tmp1.loc[:, 'econ_attr_'].value_counts()
single_attr_econ_vs_nonecon.index = single_attr_econ_vs_nonecon.index.map({True: 'economic', False: 'non-economic'})
del tmp1

tmp = tmp[~tmp.loc[:, 'single_attr_']].copy()
tmp.loc[:, 'only_within_'] = (tmp[econ_attrs].sum(axis=1) >= 2) & (tmp[nonecon_attrs].sum(axis=1) == 0) | (tmp[econ_attrs].sum(axis=1) == 0) & (tmp[nonecon_attrs].sum(axis=1) >= 2)
only_within_vs_cross = tmp.loc[:, 'only_within_'].value_counts(normalize=True)
only_within_vs_cross = only_within_vs_cross.to_frame()
only_within_vs_cross.loc[:, 'count'] = tmp.loc[:, 'only_within_'].value_counts()
only_within_vs_cross.index = only_within_vs_cross.index.map({True: 'only within dimension', False: 'cross dimension'})

tmp1 = tmp[tmp.loc[:, 'only_within_']].copy()
tmp1.loc[:, 'only_within_econ_vs_nonecon'] = tmp1[econ_attrs].sum(axis=1) >= 2
only_within_econ_vs_nonecon = tmp1.loc[:, 'only_within_econ_vs_nonecon'].value_counts(normalize=True)
only_within_econ_vs_nonecon = only_within_econ_vs_nonecon.to_frame()
only_within_econ_vs_nonecon.loc[:, 'count'] = tmp1.loc[:, 'only_within_econ_vs_nonecon'].value_counts()
only_within_econ_vs_nonecon.index = only_within_econ_vs_nonecon.index.map({True: 'economic', False:
'non-economic'})
del tmp1




# FIGURE 2
fig_cap = r'Sankey diagram showing the breakdown of social group mentions by attribute prevalence and attribute combinations. The leftmost column shows the total number of mentions extracted in our manifesto sentences corpus, which are then split into "universal" mentions (no predicted attributes) and mentions with at least one predicted attribute. The latter are further divided into single-attribute and multi-attribute mentions, with multi-attribute mentions split into those with only within-dimension combinations and those with cross-dimension combinations. The rightmost column shows the breakdown of within-dimension combinations into economic and non-economic attributes.'

# Color schemes for each dimension
# TODO: consider moving to src/reporting.py
econ_base = '#66c2a5' # mcolors.to_hex(plt.cm.Set2(0))
nonecon_base = '#fc8d62' # mcolors.to_hex(plt.cm.Set2(1))

econ_colors = {'single': econ_base}
econ_colors['within'] = intensify_color(econ_base, sat_factor=2, val_factor=0.85)  # more saturated + darker
nonecon_colors = {'single': nonecon_base}
nonecon_colors['within'] = intensify_color(nonecon_base, sat_factor=2, val_factor=0.85)  # more saturated + darker
econ_colors['cross'] = nonecon_colors['cross'] = blend_hex_colors(econ_base, nonecon_base, 0.5)  # more yellow

econ_colors['both'] = blend_hex_colors(econ_colors['within'], econ_colors['cross'], 0.5)
nonecon_colors['both'] = blend_hex_colors(nonecon_colors['within'], nonecon_colors['cross'], 0.5)

# ── extract counts from previously computed dataframes ───────────────────────
total_n           = universal_vs_others['count'].sum()
n_universal       = universal_vs_others.loc['universal', 'count']
n_has_attrs       = universal_vs_others.loc['has attributes', 'count']
n_single          = single_vs_multi_attrs.loc['single-attribute', 'count']
n_multi           = single_vs_multi_attrs.loc['multi-attribute', 'count']
n_single_econ     = single_attr_econ_vs_nonecon.loc['economic', 'count']
n_single_nonecon  = single_attr_econ_vs_nonecon.loc['non-economic', 'count']
n_within          = only_within_vs_cross.loc['only within dimension', 'count']
n_cross           = only_within_vs_cross.loc['cross dimension', 'count']
n_within_econ     = only_within_econ_vs_nonecon.loc['economic', 'count']
n_within_nonecon  = only_within_econ_vs_nonecon.loc['non-economic', 'count']

# ── helpers ──────────────────────────────────────────────────────────────────
def _flow(ax, x0r, x1l, s_bot, s_top, t_bot, t_top, color, alpha=0.35):
    """Filled cubic-Bezier connector between two node slices."""
    mid = (x0r + x1l) / 2
    verts = [
        (x0r, s_bot), (mid, s_bot), (mid, t_bot), (x1l, t_bot),
        (x1l, t_top), (mid, t_top), (mid, s_top), (x0r, s_top),
        (x0r, s_bot),
    ]
    codes = [
        PatchPath.MOVETO,
        PatchPath.CURVE4, PatchPath.CURVE4, PatchPath.CURVE4,
        PatchPath.LINETO,
        PatchPath.CURVE4, PatchPath.CURVE4, PatchPath.CURVE4,
        PatchPath.CLOSEPOLY,
    ]
    ax.add_patch(patches.PathPatch(
        PatchPath(verts, codes), facecolor=color, edgecolor='none', alpha=alpha, zorder=1
    ))


def _rect(ax, x, yb, h, w, color, alpha=0.85):
    ax.add_patch(patches.Rectangle(
        (x, yb), w, h,
        facecolor=color, edgecolor='white', linewidth=0.7, alpha=alpha, zorder=2
    ))


def _layout(counts, col, col_x, y_start, scale, gap=0.012):
    """Stack nodes; returns list of (x, ybot, ytop)."""
    out, y = [], y_start
    for i, c in enumerate(counts):
        h = c * scale
        out.append((col_x[col], y, y + h))
        y += h + (gap if i < len(counts) - 1 else 0)
    return out


# ── layout parameters ────────────────────────────────────────────────────────
S       = 1.0 / total_n          # scale: 1 unit = total_n mentions
NW      = 0.14                   # node width
G       = 0.012                  # gap between stacked nodes
hscale_ = 0.8                   # horizontal spacing between columns
levels_ = 5                      # number of columns
CX      = list(np.arange(levels_) * hscale_)  # column x-positions

# ── node positions ───────────────────────────────────────────────────────────
L0  = _layout([total_n],                         0, CX, 0,          S, G)
L1  = _layout([n_universal, n_has_attrs],         1, CX, 0,          S, G)
L2  = _layout([n_single, n_multi],                2, CX, L1[1][1],  S, G)   # anchor at has_attrs bottom
L3a = _layout([n_single_econ, n_single_nonecon],  3, CX, L2[0][1],  S, G)   # anchor at single bottom
L3b = _layout([n_within, n_cross],                3, CX, L2[1][1],  S, G)   # anchor at multi bottom
L4  = _layout([n_within_econ, n_within_nonecon],  4, CX, L3b[0][1], S, G)   # anchor at within bottom

# unpack
(x0,  y0b,  y0t ) = L0[0]
(x1u, y1ub, y1ut) = L1[0]   # universal
(x1a, y1ab, y1at) = L1[1]   # has_attrs
(x2s, y2sb, y2st) = L2[0]   # single
(x2m, y2mb, y2mt) = L2[1]   # multi
(x3e, y3eb, y3et) = L3a[0]  # single econ
(x3n, y3nb, y3nt) = L3a[1]  # single nonecon
(x3w, y3wb, y3wt) = L3b[0]  # within
(x3c, y3cb, y3ct) = L3b[1]  # cross
(x4e, y4eb, y4et) = L4[0]   # within econ
(x4n, y4nb, y4nt) = L4[1]   # within nonecon

# ── node colours ─────────────────────────────────────────────────────────────
NC = dict(
    all='#888888', univ='#bbbbbb', attr='#555555',
    single='#888888', multi='#333333',
    se=econ_base, sn=nonecon_base,
    within='#888888', cross=econ_colors['cross'],
    we=econ_colors['within'], wn=nonecon_colors['within'],
)

# ── draw ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4), dpi=200)

# flows (source slice ↔ target node)
_flow(ax, x0+NW,  x1u, y0b,                 y0b + n_universal*S, y1ub, y1ut, NC['univ'])
_flow(ax, x0+NW,  x1a, y0b + n_universal*S, y0t,                 y1ab, y1at, NC['attr'])
_flow(ax, x1a+NW, x2s, y1ab,                y1ab + n_single*S,   y2sb, y2st, NC['single'])
_flow(ax, x1a+NW, x2m, y1ab + n_single*S,   y1at,                y2mb, y2mt, NC['multi'])
_flow(ax, x2s+NW, x3e, y2sb,                y2sb + n_single_econ*S, y3eb, y3et, NC['se'])
_flow(ax, x2s+NW, x3n, y2sb + n_single_econ*S, y2st,            y3nb, y3nt, NC['sn'])
_flow(ax, x2m+NW, x3w, y2mb,                y2mb + n_within*S,   y3wb, y3wt, NC['within'])
_flow(ax, x2m+NW, x3c, y2mb + n_within*S,   y2mt,                y3cb, y3ct, NC['cross'])
_flow(ax, x3w+NW, x4e, y3wb,                y3wb + n_within_econ*S, y4eb, y4et, NC['we'])
_flow(ax, x3w+NW, x4n, y3wb + n_within_econ*S, y3wt,            y4nb, y4nt, NC['wn'])

# node rectangles
for key, (x, yb, yt) in [
    ('all',    L0[0]), ('univ',   L1[0]), ('attr',   L1[1]),
    ('single', L2[0]), ('multi',  L2[1]),
    ('se',     L3a[0]), ('sn',    L3a[1]),
    ('within', L3b[0]), ('cross', L3b[1]),
    ('we',     L4[0]),  ('wn',    L4[1]),
]:
    _rect(ax, x, yb, yt - yb, NW, NC[key])

# labels (text to the right of each node, % of total)
label_specs = [
    (L0[0],   total_n,          'all mentions'),
    (L1[0],   n_universal,      'no attributes ("universal" mentions)'),
    (L1[1],   n_has_attrs,      '≥1 attribute'),
    (L2[0],   n_single,         'single-attribute'),
    (L2[1],   n_multi,          'multi-attribute'),
    (L3a[0],  n_single_econ,    'one economic attribute'),
    (L3a[1],  n_single_nonecon, 'one non-economic attribute'),
    # (L3b[0],  n_within,         'only within-dimension combination'),
    (L3b[1],  n_cross,          'cross-dimension attribute combinations'),
    (L4[0],   n_within_econ,    '≥2 economic attributes'),
    (L4[1],   n_within_nonecon, '≥2 non-economic attributes'),
]
for (x, yb, yt), cnt, lbl in label_specs:
    mid = (yb + yt) / 2
    pct = cnt / total_n * 100

    # Draw curly brace
    brace_x = x + NW + 0.005
    brace_depth = 0.03  # depth of bracket along x-axis
    brace_height = (yt - yb) * 0.99
    brace_y_start = mid - brace_height / 2
    brace_y_end = mid + brace_height / 2

    brace_color = "#868585" 
    # Draw vertical line on the right
    ax.plot([brace_x + brace_depth, brace_x + brace_depth], [brace_y_start, brace_y_end], 
            color=brace_color, linewidth=0.8, zorder=3)
    # Draw horizontal ticks extending left
    ax.plot([brace_x, brace_x + brace_depth], [brace_y_start, brace_y_start], 
            color=brace_color, linewidth=0.6, zorder=3)
    ax.plot([brace_x, brace_x + brace_depth], [brace_y_end, brace_y_end], 
            color=brace_color, linewidth=0.6, zorder=3)

    # Add text label
    ax.text(brace_x + brace_depth + 0.03, mid,
            f'{pct:.0f}% {lbl}' if pct < 100 else f'{pct:.0f}% ({lbl})',
            ha='left', va='center', fontsize=9, linespacing=1.35,
            color='#222222')

# column header labels
for col, header in enumerate(['', '"universal" or not?', 'single- or multi-attribute?', 'attribute combinations?', '']):
    if header:
        ax.text(CX[col] + NW / 2, y1at + 0.05, header,
                ha='center', va='bottom', fontsize=8, color='#666666',
                fontstyle='italic')

ax.set_xlim(-0.05, CX[-1] + NW + 0.75)
ax.set_ylim(-0.02, y1at + 0.09)
ax.axis('off')
plt.tight_layout()
# plt.show()

save_figure(fig, 'figure02.pdf', caption=fig_cap, dest=figures_path)




# FIGURE 3
fig_cap = r"Prevalence of attribute categories across all social group mentions (excluding universal mentions with no specific attributes). Bars show the share of mentions featuring each attribute, with 95\% confidence intervals. Top panel: economic attributes; bottom panel: non-economic attributes."

# Compute overall prevalence
econ_prev = compute_prevalence(df[df['universal']==False], econ_attrs)
nonecon_prev = compute_prevalence(df[df['universal']==False], nonecon_attrs)

# Map to readable names
econ_prev['attribute'] = econ_prev['attribute'].map(attribute_category_names_map)
nonecon_prev['attribute'] = nonecon_prev['attribute'].map(attribute_category_names_map)

# Prepare subplot layout with dynamic heights
n_econ = econ_prev.attribute.nunique()
n_nonecon = nonecon_prev.attribute.nunique()
r_ = 0.175
fig_height = r_ * n_econ + r_ * n_nonecon + 2
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, fig_height),
    gridspec_kw={'height_ratios': [n_econ, n_nonecon], 'hspace': 0.2},
    sharex=True
)

# Plot
plot_prevalence_bars(econ_prev.sort_values("prevalence", ascending=False), axes[0], 'Economic attributes')
plot_prevalence_bars(nonecon_prev.sort_values("prevalence", ascending=False), axes[1], 'Non-economic attributes')

# plt.tight_layout()
# plt.show()

save_figure(fig, 'figure03.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE 4
fig_cap = r"Co-occurrence patterns of attribute categories in social group mentions. Heatmap cells show the share of mentions where each focal attribute (rows) co-occurs with other attributes (columns). The first column ('mentioned alone') shows the share of mentions where the focal attribute appears without any other attributes. Top panel: economic attributes; bottom panel: non-economic attributes. Values below 0.01 are not displayed."

# Filter to mentions with ≥1 economic attribute
attr_presence = df[[c for c in econ_attrs + nonecon_attrs if c in df.columns]].apply(lambda col: binarize_column(col))
has_any_attrs = attr_presence.sum(axis=1) >= 1
df_with_attrs = df[has_any_attrs].copy()

# Compute co-occurrence breakdown
cooc_breakdown = compute_cooccurrence_breakdown(df_with_attrs, econ_attrs + nonecon_attrs)

econ_attr_names = [attribute_category_names_map[a] for a in econ_attrs]
nonecon_attr_names = [attribute_category_names_map[a] for a in nonecon_attrs]

heatmap_data = cooc_breakdown.pivot_table(
    index='focal_attr',
    columns='cooccur_with',
    values='prevalence',
    aggfunc='first'
).fillna(0)

heatmap_data.rename(index=attribute_category_names_map, columns=attribute_category_names_map, inplace=True)
heatmap_data.rename(columns={'alone': 'mentioned alone'}, inplace=True)

fig, axes = plot_heatmap(
    x=heatmap_data,
    panel_groups=(["mentioned alone"], econ_attr_names, nonecon_attr_names),
    mask_diagonal=False,
    cmin=0.01,
    cmap='RdPu',
    clims=(0, 1.0),
    clegend_title='Prevalence of co-occurrence',
)
# make first x-label bold
axes[0].get_xticklabels()[0].set_fontweight('bold')
# set y-labels for panels
axes[0].set_ylabel("economic\n", fontweight='bold')
axes[1].set_ylabel("non-economic\n", fontweight='bold')
# plt.show()

save_figure(fig, 'figure04.pdf', caption=fig_cap, dest=figures_path) 


## Socio-structural expectations



# Compute prevalence by party family
dim_fam_prev = compute_prevalence(df, ['economic', 'non-economic', 'universal'], group_by='party_family_label')

econ_fam_prev = compute_prevalence(df[df['universal']==0], econ_attrs, group_by='party_family_label')
nonecon_fam_prev = compute_prevalence(df[df['universal']==0], nonecon_attrs, group_by='party_family_label')

# Map to readable names
dim_fam_prev['attribute'] = pd.Categorical(dim_fam_prev['attribute'], categories=['non-economic', 'economic', 'universal'], ordered=True)
econ_fam_prev['attribute'] = econ_fam_prev['attribute'].map(attribute_category_names_map)
nonecon_fam_prev['attribute'] = nonecon_fam_prev['attribute'].map(attribute_category_names_map)

# Verify CI bounds
assert econ_fam_prev.query("prevalence < ci_low | prevalence > ci_high").empty
assert nonecon_fam_prev.query("prevalence < ci_low | prevalence > ci_high").empty




# FIGURE 5
fig_cap = r"Prevalence of attribute dimensions in all mentions and economic and non-economic attributes in non-universal social group mentions in parties' election manifestos by party family. Bars show share of mentions containing each attribute, with 95\% confidence intervals. \emph{Note:} Top panel shows prevalence in all mentions, while middle and bottom panels show prevalence in mentions containing at least one attribute."

all_fam_order_alt = ['Populist Radical-Right', 'Conservative', 'Social Democratic', 'Green']

# Prepare subplot layout with dynamic heights
n_econ = econ_fam_prev['attribute'].nunique()
n_nonecon = nonecon_fam_prev['attribute'].nunique()
r_ = 0.5
fig_height = r_ * 3 + r_ * n_econ + r_ * n_nonecon + 2.5
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6, fig_height*0.65),
    gridspec_kw={'height_ratios': [3, n_econ, n_nonecon]},
    sharex=True,
    dpi=300
)

# Plot using the generalized function
plot_prevalence_bars(
    dim_fam_prev,
    axes[0],
    title='Attribute dimensions',
    attribute_order=['universal', 'economic', 'non-economic'],
    hue_col='party_family_label',
    hue_order=all_fam_order_alt,
    palette=all_fam_palette,
    fontsize=5,
    xlim=(0, .6)
)

plot_prevalence_bars(
    econ_fam_prev,
    axes[1],
    'Economic attributes',
    hue_col='party_family_label',
    hue_order=all_fam_order_alt,
    palette=all_fam_palette,
    fontsize=5,
    xlim=(0, .6)
)

plot_prevalence_bars(
    nonecon_fam_prev,
    axes[2],
    'Non-economic attributes',
    hue_col='party_family_label',
    hue_order=all_fam_order_alt,
    palette=all_fam_palette,
    fontsize=5,
    xlim=(0, .6)
)

# Add shared legend below the lower plot
handles = [plt.Rectangle((0,0),1,1, fc=all_fam_palette[fam]) for fam in all_fam_order]
fig.legend(handles, all_fam_order, loc='lower center', bbox_to_anchor=(0.65, -0.06), ncol=2, frameon=False)

plt.tight_layout()
# plt.show()

save_figure(fig, 'figure05.pdf', caption=fig_cap, dest=figures_path) 




fam_df = df[df.party_family.isin(['green', 'prrp'])].copy()
# print(fam_df.party_family.value_counts())




# FIGURE 6
fig_cap = r"Most distinctive words of group mentions in selected attribute categories in Populist Radical-Right and Green parties' election manifestos. Values plotted are $z$-scores from ``fighting words'' analysis. Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

with open(figures_path / 'figure06-caption.tex', 'w') as f:
    f.write(fig_cap)

vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=stopwords,
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)
analyzer = vectorizer.build_analyzer()

tmp = fam_df.loc[fam_df.noneconomic__gender_sexuality==1, ["text", "party_family"]].copy()

fw = compute_fighting_words(
    l1=tmp.loc[tmp["party_family"]=="prrp", 'text'].tolist(),
    l2=tmp.loc[tmp["party_family"]=="green", 'text'].tolist(),
    cv=vectorizer,
)

fw_party_family_contrasts_gender_sexuality = pd.DataFrame(fw, columns=['word', 'score'])
fw_party_family_contrasts_gender_sexuality.sort_values('score', ascending=False, inplace=True)

# get top 20 terms
top_prrp = fw_party_family_contrasts_gender_sexuality.nlargest(20, 'score').sort_values('score', ascending=True)
top_green = fw_party_family_contrasts_gender_sexuality.nsmallest(20, 'score').sort_values('score', ascending=False)
top_green.score *= -1

# Create two-column layout
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for ax, top_df, panel in zip(axes, [top_prrp, top_green], ['Populist Radical-Right', 'Green']):
    ttl = f"{panel} parties"
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.barh(range(len(top_df)), top_df['score'], color=fam_col_palette[panel], zorder=2)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['word'])
    ax.set_xlabel('z-score', fontsize=11)
    ax.set_title(ttl, fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlim(0, 8)

# Invert to show positive values extending left
ax = axes[0]
ax.invert_xaxis()  
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.setp(ax.get_yticklabels(), ha='left')

plt.tight_layout()
# plt.show()

subcap = "Gender/sexuality-related social group mentions" 
save_figure(fig, 'figure06a.pdf', caption=subcap, dest=figures_path) 

vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=list(ENGLISH_STOP_WORDS),
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)
analyzer = vectorizer.build_analyzer()

tmp = fam_df.loc[fam_df.economic__occupation_profession==1, :].copy()
tmp = tmp[tmp[label_cols].sum(axis=1)==1]
tmp = tmp.loc[:, ["text", "party_family"]]

fw = compute_fighting_words(
    l1=tmp.loc[tmp["party_family"]=="prrp", 'text'].tolist(),
    l2=tmp.loc[tmp["party_family"]=="green", 'text'].tolist(),
    cv=vectorizer,
)

fw_party_family_contrasts_occupation_profession = pd.DataFrame(fw, columns=['word', 'score'])
fw_party_family_contrasts_occupation_profession.sort_values('score', ascending=False, inplace=True)

# get top 20 terms
top_prrp = fw_party_family_contrasts_occupation_profession.nlargest(20, 'score').sort_values('score', ascending=True)
top_green = fw_party_family_contrasts_occupation_profession.nsmallest(20, 'score').sort_values('score', ascending=False)
top_green.score *= -1

# Create two-column layout
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for ax, top_df, panel in zip(axes, [top_prrp, top_green], ['Populist Radical-Right', 'Green']):
    ttl = f"{panel} parties"
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.barh(range(len(top_df)), top_df['score'], color=fam_col_palette[panel], zorder=2)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['word'])
    ax.set_xlabel('z-score', fontsize=11)
    ax.set_title(ttl, fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlim(0, 10)

# Invert to show positive values extending left
ax = axes[0]
ax.invert_xaxis()  
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.setp(ax.get_yticklabels(), ha='left')

plt.tight_layout()
# plt.show()

subcap = "Occupation/profession-related social group mentions"
save_figure(fig, 'figure06b.pdf', caption=subcap, dest=figures_path) 


## Constructivist expectations



# FIGURE 7
fig_cap = r"Differences in conditional probabilities of attribute co-occurrence between Green and Populist Radical-Right (PRR) party manifestos. Heatmap cells show the difference in $\Pr(B|A)$ between Green and PRR parties (Green – PRR). Rows indicate ``attribute A'' (the conditioning attribute) and columns indicate ``attribute B'' (the outcome attribute). Positive values (green) indicate that attribute B is more likely to be mentioned given attribute A in Green party manifestos compared to PRR manifestos, while negative values (purple) indicate the opposite. \emph{Note:} Values below 0.01 in absolute value are not displayed."

# Compute for both parties
df_prrp = df.query("universal==0 and party_family=='prrp'")
df_green = df.query("universal==0 and party_family=='green'")

cpr_prrp = compute_attribute_associations(df_prrp, label_cols)["p_b_given_a"]
cpr_green = compute_attribute_associations(df_green, label_cols)["p_b_given_a"]

cpr_diffs = cpr_green - cpr_prrp

cpr_diffs.rename(index=attribute_category_names_map, columns=attribute_category_names_map, inplace=True)

r_ = 0.3
# Use plot_heatmap function
fig, axes = plot_heatmap(
    x=cpr_diffs,
    panel_groups=(econ_attr_names, nonecon_attr_names),
    mask_diagonal=True,
    clims=(-r_, +r_),
    cmin=0.01,
    clegend_title="conditional probability difference\n(PRR vs. Green parties)",
)
axes[0].set_ylabel("economic\n", fontweight='bold')
axes[1].set_ylabel("non-economic\n", fontweight='bold')
# plt.show()

save_figure(fig, 'figure07.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE 8
fig_cap = r"Most distinctive words for mentions of \emph{shared values/mentalities}-based references to `society' in Populist Radical-Right and Green parties' election manifestos.  Values plotted are $z$-scores from ``fighting words'' analysis. Values above ±1.96 (vertical dashed line) can be considered significantly distinctive."

vectorizer = CountVectorizer(
    tokenizer=lemma_tokenizer,
    token_pattern=None,
    lowercase=False,  # handled in tokenizer
    stop_words=stopwords,
    ngram_range=(1, 3), 
    max_df=0.8,
    min_df=5
)
analyzer = vectorizer.build_analyzer()

tmp = fam_df.loc[fam_df.noneconomic__shared_values_mentalities==1, ["text", "party_family"]].copy()
tmp = tmp[tmp.text.str.contains('ociet', na=False)]

fw = compute_fighting_words(
    l1=tmp.loc[tmp["party_family"]=="prrp", 'text'].tolist(),
    l2=tmp.loc[tmp["party_family"]=="green", 'text'].tolist(),
    cv=vectorizer,
)

fw_party_family_contrasts_shared_values_mentalities = pd.DataFrame(fw, columns=['word', 'score'])
fw_party_family_contrasts_shared_values_mentalities.sort_values('score', ascending=False, inplace=True)

# get top 20 terms
top_prrp = fw_party_family_contrasts_shared_values_mentalities.nlargest(20, 'score').sort_values('score', ascending=True)
top_green = fw_party_family_contrasts_shared_values_mentalities.nsmallest(20, 'score').sort_values('score', ascending=False)
top_green.score *= -1

# Create two-column layout
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

for ax, top_df, panel in zip(axes, [top_prrp, top_green], ['Populist Radical-Right', 'Green']):
    ttl = f"{panel} parties"
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax.barh(range(len(top_df)), top_df['score'], color=fam_col_palette[panel], zorder=2)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['word'])
    ax.set_xlabel('z-score', fontsize=11)
    ax.set_title(ttl, fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlim(0, 5)

# Invert to show positive values extending left
ax = axes[0]
ax.invert_xaxis()  
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
plt.setp(ax.get_yticklabels(), ha='left')

plt.tight_layout()
# plt.show()

save_figure(fig, 'figure08.pdf', caption=fig_cap, dest=figures_path) 


## Group evaluation and deservingness



# GENDER/SEXUALITY
dest = intermediate_path / 'gender_sexuality_mentions_stance.pkl'
horne_gender_sexuality_cats = [
    'Women',
    'Men',
    'Lgbtqi',
]
if not dest.exists():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from inference.nli_stance import classify_stance_batch
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

    # Load model and tokenizer
    model_name = "rwillh11/mdeberta_NLI_stance_NoContext"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", local_files_only=True)

    fp = intermediate_path / 'horne_predictions_nonecon_gender-sexuality_attribute_sample.pkl'
    horne_predictions_nonecon_gender_attribute_sample = pd.read_pickle(fp)

    tmp = horne_predictions_nonecon_gender_attribute_sample.copy()
    tmp = tmp[tmp.party_family.isin(["prrp", "green"])]
    tmp["party_family"] = tmp.party_family.map(family_map)
    tmp = tmp[tmp[horne_gender_sexuality_cats].any(axis=1)].reset_index(drop=True)

    # NOTE: within category multi-label very unlikely (47 vs. 7924) ...
    tmp[horne_gender_sexuality_cats].sum(axis=1).value_counts()

    # ... so we subset to single-label mentions to focus the analysis
    tmp = tmp[tmp[horne_gender_sexuality_cats].sum(axis=1)==1]

    tmp["group_category"] = tmp[horne_gender_sexuality_cats].apply(lambda x: horne_gender_sexuality_cats[x.argmax()], axis=1)
    tmp["group_category"] = pd.Categorical(tmp["group_category"], categories=horne_gender_sexuality_cats, ordered=True)

    tmp = tmp[["country_iso3c", "party_id", "party_family", "date", "sentence_id", "sentence_text", "mention_id", "text", "group_category"]]
    tmp.reset_index(drop=True, inplace=True)

    # predict stance
    stance_df = classify_stance_batch(tmp[["sentence_text", "text"]].values.tolist(), model, tokenizer, batch_size=128, verbose=True)

    gender_sexuality_mentions_stance_df = pd.concat([tmp, stance_df[['positive', 'neutral', 'negative', 'predicted_stance']]], axis=1)

    gender_sexuality_mentions_stance_df.to_pickle(dest)
else:
    gender_sexuality_mentions_stance_df = pd.read_pickle(dest)




# FIGURE 9
fig_cap = r"Stance distribution towards \emph{gender/sexuality}-related groups by party family. Top panel shows overall distribution across all gender/sexuality groups. Subsequent panels show distributions for specific group categories (LGBTQI, Men, Women), with bars comparing stance distributions between Populist Radical-Right and Green parties."

stance_colors = {'positive': "#45b4ad", 'neutral': '#999999', 'negative': "#c827d7"}

def plot_stance_panel(ax, data, stance_colors, title, show_labels=False):
    """
    Plot stance distribution bars on a single panel.

    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pd.DataFrame
        DataFrame with party families as rows and stance columns (positive, neutral, negative)
    stance_colors : dict
        Mapping of stance to color
    title : str
        Panel title
    show_labels : bool
        Whether to show x-axis tick labels
    """
    # Create stacked bars
    data.plot.barh(
        stacked=True, 
        color=[stance_colors[c] for c in data.columns],
        ax=ax, 
        width=0.6, 
        edgecolor='white', 
        alpha=0.8,
        linewidth=1,
        legend=False
    )

    # Add percentage labels
    for i, family in enumerate(data.index):
        left = 0
        for stance in data.columns:
            val = data.loc[family, stance]
            if val > 0.05:
                ax.text(left + val/2, i, f'{val:.0%}', 
                       ha='center', va='center',
                       fontsize=8, color='white')
            left += val

    # Customize panel
    ax.set_xlim(0, 1.0)
    ax.set_ylabel('')
    ax.set_title(title, fontweight='bold', fontsize=10, loc='left')
    if not show_labels:
        ax.tick_params(labelbottom=False)
    ax.grid()

# Prepare overall stance distribution
stance_shares_df = (
    gender_sexuality_mentions_stance_df
    .groupby("party_family", observed=False)["predicted_stance"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
)

# Prepare subcategory stance distributions
stance_shares_by_group_df = (
    gender_sexuality_mentions_stance_df
    .groupby(["party_family", "group_category"], observed=False)["predicted_stance"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
)

# Plot settings - colorblind-friendly palette
stance_colors = {'positive': "#45b4ad", 'neutral': '#999999', 'negative': "#c827d7"}
n_groups = len(horne_gender_sexuality_cats)

# Create subplot grid: 1 overall + 3 subcategories
fig, axes = plt.subplots(n_groups + 1, 1, figsize=(6, 1.1 * (n_groups + 1)), sharex=True)

# Plot overall distribution
plot_stance_panel(axes[0], stance_shares_df, stance_colors, 'overall', show_labels=False)

# Plot subcategory distributions (in reverse order)
for idx, group_cat in enumerate(horne_gender_sexuality_cats):
    ax = axes[idx + 1]
    group_data = stance_shares_by_group_df.xs(group_cat, level='group_category')
    show_labels = (idx == len(horne_gender_sexuality_cats) - 1)  # Only show labels on last panel
    plot_stance_panel(ax, group_data, stance_colors, group_cat, show_labels=show_labels)
    ax.set_title(group_cat.upper() if group_cat=='Lgbtqi' else group_cat, fontstyle='italic', fontsize=10, loc='left')

# Add legend above the top panel
axes[-1].legend(
    title=None,
    labels=['Positive', 'Neutral', 'Negative'],
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=3,
    frameon=False,
    fontsize=9
)

plt.tight_layout()
# plt.show()

save_figure(fig, 'figure09.pdf', caption=fig_cap, dest=figures_path) 


# Additional results



fam_ts = df.copy()
fam_ts = fam_ts[fam_ts['universal']==0]

fam_ts['decade'] = (fam_ts['year'] // 10)*10




plot_data = compute_prevalence(fam_ts, attribute_category_names_map, group_by=['party_family_label', 'decade'])
plot_data['attribute_label'] = plot_data['attribute'].map(attribute_category_names_map)

upper_y = plot_data.groupby('attribute')['ci_high'].max()
upper_y = round(upper_y * 1.075, 2)
upper_y = dict(upper_y)




# FIGURE D1
fig_cap = r"Temporal trends in the prevalence of economic attributes in social group mentions by PRR and Green parties across decades. Each panel shows one economic attribute category, with error bars representing 95\% confidence intervals. Lines show the share of mentions containing each attribute over time."

these = econ_attrs


heights = [upper_y[a] for a in these]
heights /= sum(heights)

fig, axes = plt.subplots(len(these), 1, figsize=(6, 1.5 * len(these)), sharex=False, height_ratios=heights, gridspec_kw={'hspace': 4/3})
for attr_to_plot, ax in zip(these, axes):

    attr_ts = plot_data[plot_data['attribute'] == attr_to_plot].copy()

    for fam, sub in attr_ts.groupby('party_family_label'):
        if sub.empty:
            continue
        sub.sort_values('decade', inplace=True)
        yerr = np.vstack([sub['prevalence'] - sub['ci_low'], sub['ci_high'] - sub['prevalence']])
        ax.errorbar(
            sub['decade'],
            sub['prevalence'],
            yerr=yerr,
            fmt='o-',
            # color=fam_col_palette[fam],
            color=all_fam_palette[fam],
            label=fam,
            capsize=0,
            linewidth=1.5,
            markersize=5,
        )
    ax.set_ylim(0, round(upper_y[attr_to_plot] * 1.15, 2))
    ax.set_title(attribute_category_names_map[attr_to_plot], fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    ax.set_ylabel("Prevalence")
    ax.set_xlabel('Decade')


# add a manual legend below the last plot (using only dot, not line)
handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=fam) for fam, col in all_fam_palette.items()]
fig.legend(handles, all_fam_palette.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

fig.tight_layout()

save_figure(fig, 'figureD01.pdf', caption=fig_cap, dest=figures_path) 




# FIGURE D2
fig_cap = r"Temporal trends in the prevalence of economic attributes in social group mentions by PRR and Green parties across decades. Each panel shows one economic attribute category, with error bars representing 95\% confidence intervals. Lines show the share of mentions containing each attribute over time."

these = [a for a in nonecon_attrs if a not in ("noneconomic__ethnicity", "noneconomic__religion", "noneconomic__place_location")]

heights = [upper_y[a] for a in these]
heights /= sum(heights)

fig, axes = plt.subplots(len(these), 1, figsize=(6, 1.5 * len(these)), sharex=False, height_ratios=heights, gridspec_kw={'hspace': 4/3})
for attr_to_plot, ax in zip(these, axes):

    attr_ts = plot_data[plot_data['attribute'] == attr_to_plot].copy()


    for fam, sub in attr_ts.groupby('party_family_label'):
        sub = sub.query("prevalence >= 0.01").copy()
        if sub.empty:
            continue
        sub.sort_values('decade', inplace=True)
        yerr = np.vstack([sub['prevalence'] - sub['ci_low'], sub['ci_high'] - sub['prevalence']])
        ax.errorbar(
            sub['decade'],
            sub['prevalence'],
            yerr=yerr,
            fmt='o-',
            # color=fam_col_palette[fam],
            color=all_fam_palette[fam],
            label=fam,
            capsize=0,
            linewidth=1.5,
            markersize=5,
        )
    ax.set_ylim(0, round(upper_y[attr_to_plot] * 1.15, 2))
    ax.set_title(attribute_category_names_map[attr_to_plot], fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    ax.set_ylabel("Prevalence")
ax.set_xlabel('Decade')


handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=fam) for fam, col in all_fam_palette.items()]
fig.legend(handles, all_fam_palette.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)

fig.tight_layout()

save_figure(fig, 'figureD02.pdf', caption=fig_cap, dest=figures_path) 


### Representativeness of our mainstream party selection



# TODO: load from file
fp = data_path / 'labeled_mentions_cmp_translations_with_party_metadata.feather'
df_cmp = pd.read_feather(fp)




df_cmp_sd = df_cmp.query("party_family=='sd'")

# Compute prevalence by party family
dim_sd_prev = compute_prevalence(df_cmp_sd, ['economic', 'non-economic', 'universal'], group_by='in_sample')

econ_sd_prev = compute_prevalence(df_cmp_sd[df_cmp_sd['universal']==0], econ_attrs, group_by='in_sample')
nonecon_sd_prev = compute_prevalence(df_cmp_sd[df_cmp_sd['universal']==0], nonecon_attrs, group_by='in_sample')

# Map to readable names
dim_sd_prev['attribute'] = pd.Categorical(dim_sd_prev['attribute'], categories=['non-economic', 'economic', 'universal'], ordered=True)
econ_sd_prev['attribute'] = econ_sd_prev['attribute'].map(attribute_category_names_map)
nonecon_sd_prev['attribute'] = nonecon_sd_prev['attribute'].map(attribute_category_names_map)




df_cmp_con = df_cmp.query("party_family=='con'")

# Compute prevalence by party family
dim_con_prev = compute_prevalence(df_cmp_con, ['economic', 'non-economic', 'universal'], group_by='in_sample')

econ_con_prev = compute_prevalence(df_cmp_con[df_cmp_con['universal']==0], econ_attrs, group_by='in_sample')
nonecon_con_prev = compute_prevalence(df_cmp_con[df_cmp_con['universal']==0], nonecon_attrs, group_by='in_sample')

# Map to readable names
dim_con_prev['attribute'] = pd.Categorical(dim_con_prev['attribute'], categories=['non-economic', 'economic', 'universal'], ordered=True)
econ_con_prev['attribute'] = econ_con_prev['attribute'].map(attribute_category_names_map)
nonecon_con_prev['attribute'] = nonecon_con_prev['attribute'].map(attribute_category_names_map)




# FIGURE D3
fig_cap = r"Prevalence of attribute dimensions in all mentions and economic and non-economic attributes in non-universal social group mentions in parties' election manifestos by party family. Bars show share of mentions containing each attribute, with 95\% confidence intervals. \emph{Note:} Top panel shows prevalence in all mentions, while middle and bottom panels show prevalence in mentions containing at least one attribute."

in_sample_order = ['in sample', 'not in sample']
in_sample_palette = {'in sample': "#4591b4", 'not in sample': "#d77627"}

# Prepare subplot layout with dynamic heights
n_econ = econ_sd_prev['attribute'].nunique()
n_nonecon = nonecon_sd_prev['attribute'].nunique()
r_ = 0.5
fig_height = r_ * 3 + r_ * n_econ + r_ * n_nonecon + 2.5
fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(6*1.5, fig_height*0.97),
    gridspec_kw={'height_ratios': [3, n_econ, n_nonecon]},
    sharex=True,
    dpi=100
)

sub_dfs = {
    'Social democratic parties': dim_sd_prev,
    'Conservative parties': dim_con_prev
}

for ax, (ttl, subdf) in zip(axes[0], sub_dfs.items()):
    plot_prevalence_bars(
        subdf,
        ax,
        title=ttl,
        attribute_order=['universal', 'economic', 'non-economic'],
        hue_col='in_sample',
        hue_order=in_sample_order,
        palette=in_sample_palette,
        xlim=(0, .6)
    )
axes[0][0].set_ylabel('Attribute dimensions\n', fontweight='bold')
axes[0][1].set_yticklabels([])


sub_dfs = [econ_sd_prev, econ_con_prev]
for ax, subdf in zip(axes[1], sub_dfs):
    plot_prevalence_bars(
        subdf,
        ax,
        title=None,
        hue_col='in_sample',
        hue_order=in_sample_order,
        palette=in_sample_palette,
        xlim=(0, .6)
    )
axes[1][0].set_ylabel('Economic attributes', fontweight='bold')
axes[1][1].set_yticklabels([])

sub_dfs = [nonecon_sd_prev, nonecon_con_prev]
for ax, subdf in zip(axes[2], sub_dfs):
    plot_prevalence_bars(
        subdf,
        ax,
        title=None,
        hue_col='in_sample',
        hue_order=in_sample_order,
        palette=in_sample_palette,
        xlim=(0, .6)
    )
axes[2][0].set_ylabel('Non-economic attributes', fontweight='bold')
axes[2][1].set_yticklabels([])

# Add shared legend below the lower plot
handles = [plt.Rectangle((0,0),1,1, fc=in_sample_palette[fam]) for fam in in_sample_order]
fig.legend(handles, in_sample_order, loc='lower center', bbox_to_anchor=(0.65, -0.06), ncol=2, frameon=False)

plt.tight_layout()
# plt.show()

save_figure(fig, 'figureD03.pdf', caption=fig_cap, dest=figures_path) 


### Conditional probabilities of attribute category co-occurrences



attr_presence = df[[c for c in econ_attrs + nonecon_attrs if c in df.columns]].apply(lambda col: binarize_column(col))

has_any_attrs = attr_presence.sum(axis=1) >= 1
df_with_attrs = df[has_any_attrs].copy()

# Compute associations for all attributes (within and across dimensions)
assoc_all = compute_attribute_associations(df_with_attrs, econ_attrs + nonecon_attrs)

# Extract key measures
cpr_all = assoc_all["p_b_given_a"]
# pmi_all = assoc_all["pmi"]  # PMI
# ppmi_all = assoc_all["ppmi"]  # Positive PMI (co-occurrence above chance)
# npmi_all = assoc_all["npmi"]  # Normalized PMI (range [-1, 1])




# FIGURE D4
fig_cap = r'Conditional probabilities of attribute co-occurrence in social group mentions. Heatmap cells show $Pr(B|A)$, the probability of mentioning attribute B given that attribute A is mentioned. Rows indicate "attribute A" (the conditioning attribute) and columns indicate "attribute B" (the outcome attribute). \emph{Note:} Values below 0.01 are not displayed.'

cpr_all_renamed = cpr_all.rename(index=attribute_category_names_map, columns=attribute_category_names_map)

fig, axes = plot_heatmap(
    x=cpr_all_renamed,
    panel_groups=(econ_attr_names, nonecon_attr_names),
    cluster_rows=False,
    cluster_cols=False,
    cmin=0.01,
    cmap='RdPu', #cmap='YlOrRd',
    clims=(0, 1),
    clegend_title='conditional probability'
)
axes[0].set_ylabel("economic\n", fontweight='bold')
axes[1].set_ylabel("non-economic\n", fontweight='bold')
# plt.show()

save_figure(fig, 'figureD04.pdf', caption=fig_cap, dest=figures_path) 




# TALE D1
tbl_cap = r"Top attribute combinations by conditional probability $\Pr(b | a)$, that is, the probability of attribute $b$ being present in a mention given that attribute $a$ is present. Only combinations with $\Pr(b | a) > 0.1$ are included. Example mentions are randomly sampled from mentions that feature both attributes and no other attributes (i.e., mentions that only feature these two attributes)."

# pivot longer
cpr_values = cpr_all.copy()
# set upper triangle and diagonal to NaN to avoid duplicates
cpr_values.values[np.triu_indices_from(cpr_values.values)] = np.nan
cpr_values = cpr_values.reset_index().melt(id_vars='attr_a', var_name='attr_b', value_name='value')
cpr_values.dropna(subset=['value'], inplace=True)
cpr_values.sort_values('value', ascending=False, inplace=True)

# Create DataFrame with top 10 nPMI combinations and example mentions
top_cpr_examples = []

for a, b, v in cpr_values.query("value > 0.1").head(10).itertuples(index=False):

    # Find mentions where both attributes are present
    idxs = df[[a, b]].apply(lambda col: binarize_column(col), axis=0).sum(axis=1) == 2
    exclude = df.loc[idxs, label_cols].sum(axis=1) != 2
    idxs = idxs & ~exclude
    n_examples = min(5, idxs.sum())

    # Sample random examples
    examples = df.loc[idxs].sample(n=n_examples, random_state=42)['text'].tolist()

    # Create row
    example_row = {
        'attr_a': attribute_category_names_map.get(a, a),
        'attr_b': attribute_category_names_map.get(b, b),
        'cpr': v,
        'ex1': examples[0],
        'ex2': examples[1],
        'ex3': examples[2],
        'ex4': examples[3],
        'ex5': examples[4]
    }

    top_cpr_examples.append(example_row)

df_top_cpr_examples = pd.DataFrame(top_cpr_examples)
df_top_cpr_examples.columns=['attribute a', 'attribute b', 'Pr(b | a)'] + [f'example {i}' for i in range(1, 6)]

tab = latex_table(df_top_cpr_examples, caption=tbl_cap, landscape=True, column_format="p{3cm} L{3cm} r L{2.5cm} L{2.5cm} L{2.5cm} L{2.5cm} L{2.5cm}", label='tableD01')
save_table(tab, 'tableD01.tex', dest=tables_path) 

