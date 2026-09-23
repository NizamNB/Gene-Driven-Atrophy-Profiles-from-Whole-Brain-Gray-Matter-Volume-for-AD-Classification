# ============================================================
# NMF GAP MATRIX → SIGNIFICANT COMPONENTS → GSEA → PPI
#
# Gene selection per significant component:
#   Criterion 1: H loading > mean + LOADING_SD*SD
#                (component-specific genes only)
#   Criterion 2: Welch t-test on RAW (unweighted) GAP scores
#                AD vs CN within pool → BH-FDR q<GENE_FDR_ALPHA
#                Fallback: nominal p<0.05 if FDR finds nothing
#   Hard cap   : MAX_SIG_GENES = 500
#
# The two criteria are INDEPENDENT:
#   C1 asks: does this gene define the component?
#   C2 asks: is this gene different between AD and CN?
#   Only genes passing BOTH are used for GSEA + PPI.
#
# PPI: ALL identified genes per component (STRING limit=0)
#      Combined PPI across all significant components
#
# GO plots per component:
#   A) Dot plot per library
#   B) Bar chart — top terms across all libraries
#   C) Bubble summary
#   D) Enrichment heatmap
#   E) Combined multi-panel summary
#
# All outputs → nmf_results/nmf_sig/
#
# CHANGES FROM THE all_gene_gap / k=27 VERSION:
#   - GAP_DIR points at allgene_OASIS_studyspecific
#   - Step 1 loads gap_matrix.csv OR gap_matrix_full.parquet,
#     whichever exists, and auto-detects the group/subject
#     metadata columns instead of assuming fixed names (the
#     same robust pattern used in the k-selection scripts,
#     since this dataset's subject column and file format
#     differ from the earlier all_gene_gap run)
#   - N_COMPONENTS = 45, from the corrected mean-centered
#     scree/cumulative-variance elbow (80% cutoff). CONFIRM
#     this with the cophenetic stability sweep before treating
#     it as final — this script does not itself check stability.
# ============================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import gseapy as gp
from scipy import stats
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
GAP_DIR        = r'C:\Users\StujenskeLab\Documents\NAN151_workspace\Alzheimer\allgene_OASIS_studyspecific'
GAP_CSV        = os.path.join(GAP_DIR, 'gap_matrix.csv')
GAP_PARQUET    = os.path.join(GAP_DIR, 'gap_matrix_full.parquet')
BASE_OUT       = os.path.join(GAP_DIR, 'nmf_results')
OUT_DIR        = os.path.join(BASE_OUT, 'nmf_sig')
os.makedirs(OUT_DIR, exist_ok=True)

N_COMPONENTS   = 45      # from corrected scree/cumulative-variance elbow —
                         # CONFIRM with cophenetic stability before finalising
COMP_FDR_ALPHA = 0.05    # component-level FDR
GENE_FDR_ALPHA = 0.05    # gene-level FDR within pool
LOADING_SD     = 2.5     # loading threshold: mean + 2.5*SD
MAX_SIG_GENES  = 500     # hard cap — take top N by |Cohen's d|
RANDOM_STATE   = 42
HUB_MIN_DEGREE = 5
STRING_SCORE   = 400
RED            = '#D85A30'
BLUE           = '#185FA5'

LIB_COLORS = {
    'GO_Biological_Process_2023': '#1D9E75',
    'GO_Cellular_Component_2023': '#534AB7',
    'GO_Molecular_Function_2023': '#D85A30',
    'KEGG_2021_Human':            '#185FA5',
}
LIB_SHORT = {
    'GO_Biological_Process_2023': 'GO BP',
    'GO_Cellular_Component_2023': 'GO CC',
    'GO_Molecular_Function_2023': 'GO MF',
    'KEGG_2021_Human':            'KEGG',
}
GO_LIBS = list(LIB_COLORS.keys())

# ============================================================
# HELPERS
# ============================================================

def resolve_symbols(gene_list, batch_size=1000):
    placeholders = {}
    real_genes   = []
    for g in gene_list:
        parts = str(g).split('_')
        if (str(g).startswith('GENE_') and
                len(parts) == 2 and parts[1].isdigit()):
            placeholders[g] = int(parts[1])
        else:
            real_genes.append(g)
    if not placeholders:
        return list(gene_list), {}
    sym_map = {}
    ids     = list(placeholders.values())
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        try:
            resp = requests.post(
                'https://mygene.info/v3/gene',
                json={'ids':     [str(x) for x in batch],
                      'fields':  'symbol,name',
                      'species': 'human'},
                timeout=30)
            resp.raise_for_status()
            for rec in resp.json():
                eid  = rec.get('_id')
                sym  = rec.get('symbol')
                name = rec.get('name', '')
                if eid and sym:
                    sym_map[int(eid)] = (str(sym), str(name))
        except Exception as e:
            print(f"  [mygene error] {e}")
        time.sleep(0.3)
    resolved       = list(real_genes)
    gene_to_symbol = {}
    for placeholder, eid in placeholders.items():
        result = sym_map.get(eid)
        if result:
            sym, name = result
            resolved.append(sym)
            gene_to_symbol[placeholder] = {
                'symbol': sym, 'name': name}
    print(f"  Resolved {len(gene_to_symbol):,} / "
          f"{len(placeholders):,}")
    return resolved, gene_to_symbol


def safe_padj(res_df, col='p_adj'):
    s = res_df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors='coerce').fillna(1.0)


def normalise_cols(res):
    res = res.loc[:, ~res.columns.duplicated()]
    col_map = {}
    for col in res.columns:
        cl = (col.lower().replace(' ','_')
              .replace('-','_').replace('.','_'))
        if cl == 'term':            col_map[col] = 'term'
        elif cl == 'overlap':       col_map[col] = 'overlap'
        elif cl in ('p_value','pvalue','p_val'):
            col_map[col] = 'p_value'
        elif cl in ('adjusted_p_value','adj__p_value',
                    'fdr','padj','adj_p_val'):
            col_map[col] = 'p_adj'
        elif cl in ('genes','gene_list','lead_genes'):
            col_map[col] = 'genes'
    res = res.rename(columns=col_map)
    return res.loc[:, ~res.columns.duplicated()]


def parse_overlap(res, n_genes):
    if 'overlap' in res.columns:
        def _s(x, idx):
            try:   return int(str(x).split('/')[idx])
            except: return 0
        res['n_overlap'] = res['overlap'].apply(
            lambda x: _s(x, 0))
        res['n_list']    = res['overlap'].apply(
            lambda x: _s(x, 1))
    else:
        res['n_overlap'] = 0
        res['n_list']    = n_genes
    return res


def ensure_padj(res):
    if 'p_adj' not in res.columns:
        if 'p_value' in res.columns:
            _, padj, _, _ = multipletests(
                pd.to_numeric(res['p_value'],
                              errors='coerce').fillna(1).values,
                method='fdr_bh')
            res['p_adj'] = padj
        else:
            res['p_adj'] = 1.0
    return res


def run_gsea(gene_list, library, comp_name, direction):
    if len(gene_list) < 3:
        return None
    try:
        enr = gp.enrichr(
            gene_list = gene_list,
            gene_sets = library,
            outdir    = None,
            verbose   = False,
        )
    except Exception as e:
        print(f"    [enrichr error] {library}: {e}")
        return None
    res = enr.results.copy()
    if res is None or len(res) == 0:
        return None
    res = normalise_cols(res)
    res = parse_overlap(res, len(gene_list))
    res = ensure_padj(res)
    p_adj_s = safe_padj(res, 'p_adj')
    res['p_adj']       = p_adj_s.values
    res['gene_ratio']  = (res['n_overlap'] /
                          (res['n_list'] + 1e-10))
    res['neg_log10_p'] = -np.log10(
        p_adj_s.clip(lower=1e-300).values)
    res['component']   = comp_name
    res['direction']   = direction
    res['library']     = library
    sig = res[
        (res['p_adj']     < 0.05) &
        (res['n_overlap'] >= 2)
    ].head(15).copy()
    best_p = p_adj_s.min()
    if len(sig) == 0:
        print(f"    {library}: no significant terms "
              f"(best adj-p={best_p:.4f})")
        return None
    print(f"    {library}: {len(sig)} terms  "
          f"top: {sig['term'].iloc[0][:50]}")
    return sig.sort_values('p_adj').reset_index(drop=True)


def plot_dotplot(sig_df, title, outpath):
    plot_df = sig_df.sort_values('gene_ratio')
    nmin    = plot_df['n_overlap'].min()
    nmax    = plot_df['n_overlap'].max()
    sizes   = 40 + 160*((plot_df['n_overlap']-nmin)/
                         (nmax-nmin+1))
    fig, ax = plt.subplots(
        figsize=(9, max(4, len(plot_df)*0.4+2)))
    sc = ax.scatter(
        plot_df['gene_ratio'], range(len(plot_df)),
        c=plot_df['neg_log10_p'], s=sizes,
        cmap='RdYlBu_r',
        vmin=plot_df['neg_log10_p'].min(),
        vmax=plot_df['neg_log10_p'].max(),
        edgecolors='grey', linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['term'], fontsize=8)
    ax.set_xlabel('Gene ratio  (overlap / list size)',
                  fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', color='lightgrey',
            linewidth=0.5, zorder=0)
    cb = plt.colorbar(sc, ax=ax, shrink=0.4)
    cb.set_label('-log10(adj p)', fontsize=8)
    for s, lbl in [(40,'low'),(120,'medium'),(200,'high')]:
        ax.scatter([],[],s=s,c='grey',alpha=0.6,label=lbl,
                   edgecolors='grey',linewidths=0.4)
    ax.legend(title='Gene count', fontsize=7,
              title_fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bar_chart(all_lib_results, comp_name, direction,
                   cohens_d, outpath, top_n=20):
    frames = []
    for lib, df in all_lib_results.items():
        tmp = df.copy()
        tmp['library_short'] = LIB_SHORT.get(lib, lib)
        tmp['lib_color']     = LIB_COLORS.get(lib, '#888888')
        frames.append(tmp)
    if not frames:
        return
    combined = (pd.concat(frames, ignore_index=True)
                .sort_values('p_adj')
                .drop_duplicates('term')
                .head(top_n)
                .sort_values('neg_log10_p'))
    colors = [LIB_COLORS.get(r['library'], '#888888')
              for _, r in combined.iterrows()]
    fig, ax = plt.subplots(
        figsize=(10, max(5, len(combined)*0.42+2)))
    ax.barh(combined['term'], combined['neg_log10_p'],
            color=colors, alpha=0.85,
            edgecolor='white', linewidth=0.4)
    ax.axvline(-np.log10(0.05), color='black',
               linestyle='--', linewidth=0.8,
               label='adj p = 0.05')
    col = RED if direction == 'AD' else BLUE
    ax.set_xlabel('-log10(adjusted p-value)', fontsize=10)
    ax.set_title(
        f'{comp_name}  ({direction})  d={cohens_d:.3f}\n'
        f'Top enriched terms — all GO libraries',
        fontsize=10, color=col)
    handles = [mpatches.Patch(color=v, label=LIB_SHORT[k])
               for k, v in LIB_COLORS.items()
               if k in all_lib_results]
    ax.legend(handles=handles, fontsize=8,
              title='Library', title_fontsize=8,
              loc='lower right')
    ax.grid(axis='x', color='lightgrey',
            linewidth=0.4, zorder=0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_bubble_summary(all_lib_results, comp_name,
                        direction, cohens_d, outpath,
                        top_per_lib=8):
    frames = []
    for lib, df in all_lib_results.items():
        tmp = df.head(top_per_lib).copy()
        tmp['library_short'] = LIB_SHORT.get(lib, lib)
        tmp['lib_color']     = LIB_COLORS.get(lib, '#888888')
        frames.append(tmp)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    libs     = combined['library_short'].unique()
    lib_y    = {lib: i for i, lib in enumerate(libs)}
    fig, ax  = plt.subplots(figsize=(10, 8))
    rng_b    = np.random.default_rng(42)
    for _, row in combined.iterrows():
        y    = (lib_y[row['library_short']] +
                rng_b.uniform(-0.2, 0.2))
        size = (30 + 200*(row['gene_ratio'] /
                (combined['gene_ratio'].max()+1e-10)))
        ax.scatter(row['neg_log10_p'], y,
                   s=size, color=row['lib_color'],
                   alpha=0.75, edgecolors='white',
                   linewidths=0.4, zorder=3)
        if row['neg_log10_p'] > 1.5:
            ax.annotate(
                row['term'][:35],
                xy=(row['neg_log10_p'], y),
                xytext=(6, 0),
                textcoords='offset points',
                fontsize=6, va='center',
                color='#333333')
    ax.set_yticks(range(len(libs)))
    ax.set_yticklabels(libs, fontsize=9)
    ax.set_xlabel('-log10(adjusted p-value)', fontsize=10)
    ax.axvline(-np.log10(0.05), color='black',
               linestyle='--', linewidth=0.8,
               label='adj p = 0.05')
    col = RED if direction == 'AD' else BLUE
    ax.set_title(
        f'{comp_name}  ({direction})  d={cohens_d:.3f}\n'
        f'Enrichment bubble summary  '
        f'(bubble size = gene ratio)',
        fontsize=10, color=col)
    ax.legend(fontsize=8)
    ax.grid(axis='x', color='lightgrey', linewidth=0.4)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_enrichment_heatmap(all_lib_results, comp_name,
                             direction, cohens_d, outpath,
                             top_per_lib=10):
    frames = []
    for lib, df in all_lib_results.items():
        tmp = df.head(top_per_lib)[
            ['term','neg_log10_p','library']].copy()
        frames.append(tmp)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    pivot    = combined.pivot_table(
        index='term', columns='library',
        values='neg_log10_p', fill_value=0)
    pivot.columns = [LIB_SHORT.get(c, c)
                     for c in pivot.columns]
    pivot = pivot.loc[
        pivot.max(axis=1).sort_values(
            ascending=False).index]
    if pivot.empty or pivot.shape[0] < 2:
        return
    fig, ax = plt.subplots(
        figsize=(max(6, pivot.shape[1]*2.5),
                 max(5, pivot.shape[0]*0.4+2)))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax,
                linewidths=0.3, linecolor='white',
                cbar_kws={'label':'-log10(adj p)',
                          'shrink': 0.5})
    ax.set_xlabel('GO / Pathway library', fontsize=10)
    ax.set_ylabel('Enriched term', fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=9, rotation=0)
    col = RED if direction == 'AD' else BLUE
    ax.set_title(
        f'{comp_name}  ({direction})  d={cohens_d:.3f}\n'
        f'Enrichment heatmap (-log10 adj p)',
        fontsize=10, color=col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def query_string_all(gene_list, label=''):
    genes = [g for g in gene_list if g]
    if not genes:
        return None
    print(f"  Querying STRING: {len(genes):,} genes"
          + (f"  ({label})" if label else ""))
    try:
        resp = requests.post(
            'https://string-db.org/api/json/network',
            data={
                'identifiers':     '\r'.join(genes),
                'species':         9606,
                'required_score':  STRING_SCORE,
                'limit':           0,
                'caller_identity': 'nmf_gap_analysis',
            },
            timeout=180)
        resp.raise_for_status()
        data = resp.json()
        print(f"  STRING returned {len(data)} interactions")
        return pd.DataFrame(data) if data else None
    except Exception as e:
        print(f"  [STRING error] {e}")
        return None


def build_and_save_ppi(edges_df, gene_list, title_str,
                       png_path, csv_path,
                       node_colors_map=None):
    if edges_df is None or len(edges_df) == 0:
        print(f"  No STRING edges returned")
        return None
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        a = row.get('preferredName_A',
                    row.get('stringId_A',''))
        b = row.get('preferredName_B',
                    row.get('stringId_B',''))
        s = float(row.get('score', 0))
        if a and b and a != b:
            G.add_edge(a, b, weight=s)
    keep = set(gene_list) & set(G.nodes)
    G    = G.subgraph(keep).copy()
    degs = dict(G.degree())
    print(f"  Network: {G.number_of_nodes()} proteins  "
          f"{G.number_of_edges()} interactions")
    if G.number_of_nodes() == 0:
        print(f"  No overlapping nodes")
        return None
    hubs = sorted(
        [(nd, d) for nd, d in degs.items()
         if d >= HUB_MIN_DEGREE],
        key=lambda x: x[1], reverse=True)
    hub_df = pd.DataFrame(hubs, columns=['gene','degree'])
    hub_df.to_csv(csv_path, index=False)
    print(f"  Hub genes (degree>={HUB_MIN_DEGREE}): {len(hubs)}")
    if hubs:
        print("  " + ", ".join(
            [f"{g}({d})" for g, d in hubs[:10]]))
    hub_set  = {g for g, _ in hubs}
    pos      = nx.spring_layout(G, seed=42, k=0.5)
    n_sizes  = [30 + degs.get(nd,1)*20 for nd in G.nodes]

    def node_col(nd):
        if nd in hub_set:
            return RED
        if node_colors_map and nd in node_colors_map:
            return node_colors_map[nd]
        return '#AAAAAA'

    n_colors = [node_col(nd) for nd in G.nodes]
    fig, ax  = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.3,
                           edge_color='grey', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=n_sizes,
                           node_color=n_colors,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={nd: nd for nd in G.nodes if nd in hub_set},
        font_size=7, ax=ax)
    ax.set_title(title_str, fontsize=11)
    ax.axis('off')
    ax.legend(handles=[
        mpatches.Patch(color=RED,       label='Hub gene'),
        mpatches.Patch(color='#AAAAAA', label='Other'),
    ], fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(png_path)}")
    return hub_df


# ============================================================
# STEP 1 — LOAD + ROW-NORMALISE
#   Handles csv or parquet, and auto-detects metadata columns
#   (anything not fully numeric) instead of assuming fixed
#   names — this dataset's subject-id column and file format
#   differ from the earlier all_gene_gap run.
# ============================================================
print("=" * 60)
print("STEP 1 — Loading and normalising GAP matrix")
print("=" * 60)

if os.path.exists(GAP_CSV):
    print(f"  reading {GAP_CSV}")
    df = pd.read_csv(GAP_CSV)
elif os.path.exists(GAP_PARQUET):
    print(f"  reading {GAP_PARQUET}")
    df = pd.read_parquet(GAP_PARQUET)
else:
    raise SystemExit(f"No gap_matrix.csv or gap_matrix_full.parquet found "
                     f"in {GAP_DIR}")

group_matches = [c for c in df.columns if str(c).strip().lower() == 'group']
if not group_matches:
    raise SystemExit(f"No 'group' column found. Columns present: "
                     f"{list(df.columns[:10])} ...")
group_col = group_matches[0]

candidate_cols = [c for c in df.columns if c != group_col]
numeric_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
gene_cols  = [c for c in candidate_cols if numeric_df[c].notna().all()]
meta_cols  = [c for c in candidate_cols if c not in gene_cols]
subj_col   = meta_cols[0] if meta_cols else None

print(f"  metadata columns detected: {[group_col] + meta_cols}")
n_ad = int((df[group_col] == 'AD').sum())
n_cn = int((df[group_col] == 'CN').sum())
print(f"  Subjects : {len(df)}  (AD={n_ad}, CN={n_cn})")
print(f"  Genes    : {len(gene_cols):,}")

gene_data      = numeric_df[gene_cols]
row_means      = gene_data.mean(axis=1)
gene_data_norm = gene_data.div(row_means, axis=0)
X_nmf          = gene_data_norm.clip(lower=0).values

groups      = df[group_col].values
ad_mask     = groups == 'AD'
cn_mask     = groups == 'CN'
ad_idx      = df.index[df[group_col] == 'AD']
cn_idx      = df.index[df[group_col] == 'CN']
subject_ids = (df[subj_col].values if subj_col
              else pd.Series(range(len(df))).values)
n, p        = X_nmf.shape
print(f"  Matrix   : {X_nmf.shape}")

# gene_data_norm is indexed by the default RangeIndex from df — ad_idx/cn_idx
# (also RangeIndex-based, from df.index) line up with it directly for the
# .loc calls in Step 5 below.

# ============================================================
# STEP 2 — NMF
# ============================================================
print(f"\nSTEP 2 — NMF  k={N_COMPONENTS}")
model = NMF(n_components=N_COMPONENTS, init='nndsvda',
            max_iter=2000, random_state=RANDOM_STATE,
            tol=1e-5)
W = model.fit_transform(X_nmf)
H = model.components_
print(f"  Reconstruction error : {model.reconstruction_err_:.4f}")
comp_labels = [f'NMF_{i+1}' for i in range(N_COMPONENTS)]
W_df = pd.DataFrame(W, index=subject_ids, columns=comp_labels)
W_df['group'] = groups
W_df.to_csv(os.path.join(OUT_DIR, 'nmf_W_subject_scores.csv'))
H_df = pd.DataFrame(H, index=comp_labels, columns=gene_cols)
H_df.to_csv(os.path.join(OUT_DIR, 'nmf_H_gene_loadings.csv'))
print(f"  Saved W + H matrices")

# ============================================================
# STEP 3 — TEST COMPONENTS AD vs CN
# ============================================================
print(f"\nSTEP 3 — Testing {N_COMPONENTS} components")
comp_stats = []
for i in range(N_COMPONENTS):
    sc    = W[:, i]
    ad_s  = sc[ad_mask]; cn_s = sc[cn_mask]
    t, pv = stats.ttest_ind(ad_s, cn_s, equal_var=False)
    psd   = np.sqrt(
        (ad_s.std(ddof=1)**2 + cn_s.std(ddof=1)**2) / 2)
    d = (ad_s.mean()-cn_s.mean())/psd if psd > 0 else 0.0
    comp_stats.append({
        'component': comp_labels[i], 'k': i+1,
        'AD_mean': ad_s.mean(), 'CN_mean': cn_s.mean(),
        'cohens_d': d, 't_stat': t, 'p_value': pv,
        'direction': 'AD' if d > 0 else 'CN',
    })
comp_df = pd.DataFrame(comp_stats)
reject, q_vals, _, _ = multipletests(
    comp_df['p_value'].values, alpha=COMP_FDR_ALPHA,
    method='fdr_bh')
comp_df['q_value']     = q_vals
comp_df['significant'] = reject
comp_df = comp_df.sort_values('p_value').reset_index(drop=True)
comp_df.to_csv(
    os.path.join(OUT_DIR, 'nmf_component_stats.csv'),
    index=False)
sig_comps = comp_df[comp_df['significant']]
print(f"  Significant (q<{COMP_FDR_ALPHA}): {len(sig_comps)}")
print(comp_df[['component','cohens_d','p_value',
               'q_value','significant','direction']
             ].to_string(index=False))

# ============================================================
# STEP 4 — COMPONENT SCORE PLOTS + VOLCANO
# ============================================================
print("\nSTEP 4 — Component score plots + volcano")
n_show = min(N_COMPONENTS, 15)
top_n  = comp_df.reindex(
    comp_df['cohens_d'].abs().sort_values(
        ascending=False).index).head(n_show)
ncols = 5
nrows = int(np.ceil(n_show / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(ncols*3.6, nrows*4))
axes = np.array(axes).flatten()
rng  = np.random.default_rng(RANDOM_STATE)
for ax_i, (_, row) in enumerate(top_n.iterrows()):
    k  = row['k'] - 1; sc = W[:, k]; ax = axes[ax_i]
    ad_s = sc[ad_mask]; cn_s = sc[cn_mask]
    bp = ax.boxplot([cn_s, ad_s], labels=['CN','AD'],
                    patch_artist=True, widths=0.5,
                    medianprops={'color':'black','linewidth':1.5})
    bp['boxes'][0].set_facecolor(BLUE)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(RED)
    bp['boxes'][1].set_alpha(0.7)
    ax.scatter(1+rng.uniform(-0.1,0.1,len(cn_s)), cn_s,
               color=BLUE, s=10, alpha=0.5, zorder=3)
    ax.scatter(2+rng.uniform(-0.1,0.1,len(ad_s)), ad_s,
               color=RED, s=10, alpha=0.5, zorder=3)
    sig_star = ('***' if row['q_value']<0.001 else
                '**'  if row['q_value']<0.01  else
                '*'   if row['q_value']<0.05  else 'ns')
    col = RED if row['direction']=='AD' else BLUE
    ax.set_title(
        f"{row['component']}\n"
        f"d={row['cohens_d']:.2f}  "
        f"q={row['q_value']:.3f}  {sig_star}",
        fontsize=8, color=col)
    ax.tick_params(labelsize=7)
for ax_i in range(n_show, len(axes)):
    axes[ax_i].set_visible(False)
plt.suptitle(f"NMF component scores (k={N_COMPONENTS})",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'nmf_component_scores.png'),
            dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
nlp_c = -np.log10(comp_df['p_value'].clip(lower=1e-300))
ns_cm =  ~comp_df['significant']
ad_cm =   comp_df['significant'] & (comp_df['direction']=='AD')
cn_cm =   comp_df['significant'] & (comp_df['direction']=='CN')
ax.scatter(comp_df.loc[ns_cm,'cohens_d'], nlp_c[ns_cm],
           s=60, color='#CCCCCC', alpha=0.7,
           label='Not significant')
ax.scatter(comp_df.loc[ad_cm,'cohens_d'], nlp_c[ad_cm],
           s=100, color=RED, alpha=0.9, label='Higher in AD')
ax.scatter(comp_df.loc[cn_cm,'cohens_d'], nlp_c[cn_cm],
           s=100, color=BLUE, alpha=0.9, label='Higher in CN')
if len(sig_comps):
    ax.axhline(-np.log10(sig_comps['p_value'].max()),
               color='black', linestyle='--', linewidth=1,
               label=f'FDR q={COMP_FDR_ALPHA}')
ax.axvline(0, color='black', linewidth=0.7)
for _, row in comp_df[comp_df['significant']].iterrows():
    col = RED if row['direction']=='AD' else BLUE
    ax.annotate(row['component'],
                xy=(row['cohens_d'],
                    -np.log10(max(row['p_value'],1e-300))),
                xytext=(6,0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=col)
ax.set_xlabel("Cohen's d", fontsize=11)
ax.set_ylabel('-log10(p-value)', fontsize=11)
ax.set_title(
    f'NMF component significance (k={N_COMPONENTS})',
    fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'nmf_component_volcano.png'),
            dpi=300)
plt.close()
print(f"  Saved: nmf_component_scores.png  "
      f"nmf_component_volcano.png")

analyse_comps = sig_comps.copy()
if len(analyse_comps) == 0:
    print(f"\n  No significant components at q<{COMP_FDR_ALPHA}.")
    import sys; sys.exit(0)
print(f"\n  Significant: {analyse_comps['component'].tolist()}")

# ============================================================
# STEP 5 — GENE SELECTION (INDEPENDENT TWO-CRITERION)
#
# Criterion 1: H loading > mean + LOADING_SD*SD
#   → selects genes that DEFINE the component
#   → independent of AD/CN difference
#
# Criterion 2: Welch t-test on RAW unweighted GAP scores
#   → selects genes that are DIFFERENT between AD and CN
#   → independent of loading magnitude
#   → BH-FDR applied across pool only (not 18,686)
#   → fallback: nominal p<0.05 if FDR finds nothing
#
# Hard cap: MAX_SIG_GENES by |Cohen's d|
#
# The two criteria are fully independent:
#   passing C1 does not make C2 easier to pass
#   passing C2 does not make C1 easier to pass
# ============================================================
print(f"\nSTEP 5 — Gene selection (independent two criteria)")
print(f"  C1: H loading > mean + {LOADING_SD}*SD")
print(f"  C2: raw GAP t-test BH-FDR q<{GENE_FDR_ALPHA} in pool")
print(f"  Cap: top {MAX_SIG_GENES} by |Cohen's d| if exceeded")

gene_idx_map  = {g: i for i, g in enumerate(gene_cols)}
all_sig_genes = {}

for _, comp_row in analyse_comps.iterrows():
    comp_name = comp_row['component']
    direction = comp_row['direction']
    k         = comp_row['k'] - 1
    loadings  = H[k, :]

    print(f"\n  {'='*50}")
    print(f"  {comp_name}  ({direction}  "
          f"d={comp_row['cohens_d']:.3f})")

    # ── Criterion 1: loading threshold ───────────────────────
    mean_l = loadings.mean()
    std_l  = loadings.std()
    thresh = mean_l + LOADING_SD * std_l
    pool_mask  = loadings > thresh
    pool_genes = [gene_cols[i] for i in range(p)
                  if pool_mask[i]]
    pool_loads = [loadings[i] for i in range(p)
                  if pool_mask[i]]

    print(f"  Loading: mean={mean_l:.6f}  "
          f"SD={std_l:.6f}  thresh={thresh:.6f}")
    print(f"  C1 pool: {len(pool_genes):,} genes "
          f"(top tail above mean+{LOADING_SD}SD)")

    if len(pool_genes) < 3:
        print(f"  Pool too small — skipping")
        continue

    # ── Criterion 2: t-test on RAW unweighted GAP scores ─────
    # Raw GAP scores for pool genes only
    # No multiplication by H loading — scores are independent
    ttest_rows = []
    for gene, loading in zip(pool_genes, pool_loads):
        # Raw normalised GAP scores (NOT weighted by loading)
        ad_vals = gene_data_norm.loc[ad_idx, gene].dropna().values
        cn_vals = gene_data_norm.loc[cn_idx, gene].dropna().values
        if len(ad_vals) < 3 or len(cn_vals) < 3:
            continue
        t, pv = stats.ttest_ind(
            ad_vals, cn_vals, equal_var=False)
        eff = ad_vals.mean() - cn_vals.mean()
        psd = np.sqrt(
            (ad_vals.std(ddof=1)**2 +
             cn_vals.std(ddof=1)**2) / 2)
        d = eff / psd if psd > 0 else 0.0
        ttest_rows.append({
            'gene':     gene,
            'loading':  loading,
            'AD_mean':  ad_vals.mean(),
            'CN_mean':  cn_vals.mean(),
            'effect':   eff,
            'cohens_d': d,
            'p_value':  pv,
            'gene_dir': 'AD' if eff > 0 else 'CN',
        })

    if not ttest_rows:
        print(f"  No valid t-tests")
        continue

    gene_df = pd.DataFrame(ttest_rows)

    # BH-FDR across pool only (not 18,686)
    reject, q_vals, _, _ = multipletests(
        gene_df['p_value'].values,
        alpha=GENE_FDR_ALPHA,
        method='fdr_bh')
    gene_df['q_value'] = q_vals
    gene_df['fdr_sig'] = reject

    fdr_genes     = gene_df[gene_df['fdr_sig']].copy()
    used_fallback = False
    print(f"  C2 FDR q<{GENE_FDR_ALPHA}: "
          f"{len(fdr_genes):,} / {len(gene_df):,} genes")

    if len(fdr_genes) == 0:
        fdr_genes     = gene_df[
            gene_df['p_value'] < 0.05].copy()
        used_fallback = True
        print(f"  Fallback p<0.05: {len(fdr_genes):,} genes")

    fdr_genes = fdr_genes.sort_values(
        'cohens_d',
        ascending=(direction == 'CN')).copy()
    fdr_genes['component']     = comp_name
    fdr_genes['comp_dir']      = direction
    fdr_genes['used_fallback'] = used_fallback

    # Hard cap — take top MAX_SIG_GENES by |Cohen's d|
    if len(fdr_genes) > MAX_SIG_GENES:
        print(f"  Cap: {len(fdr_genes):,} → {MAX_SIG_GENES} "
              f"by |Cohen's d|")
        fdr_genes = fdr_genes.reindex(
            fdr_genes['cohens_d'].abs().sort_values(
                ascending=False).index
        ).head(MAX_SIG_GENES).copy()

    all_sig_genes[comp_name] = fdr_genes
    print(f"  FINAL: {len(fdr_genes):,} significant genes "
          f"(C1 AND C2 — independent criteria)")

    if len(fdr_genes):
        print(f"  Top 10:")
        show = ['gene','loading','cohens_d','p_value']
        if not used_fallback:
            show.append('q_value')
        print(fdr_genes[show].head(10).to_string(index=False))

    fdr_genes.to_csv(
        os.path.join(OUT_DIR,
                     f"sig_genes_{comp_name}.csv"),
        index=False)
    print(f"  Saved: sig_genes_{comp_name}.csv")

# Loading heatmap
union_genes = []
for gdf in all_sig_genes.values():
    top30 = gdf.reindex(
        gdf['cohens_d'].abs().sort_values(
            ascending=False).index).head(30)
    union_genes.extend(top30['gene'].tolist())
union_genes = list(dict.fromkeys(union_genes))

if union_genes:
    hm_data = pd.DataFrame(index=union_genes)
    for _, row in analyse_comps.iterrows():
        k = row['k'] - 1
        hm_data[row['component']] = [
            H[k, gene_idx_map[g]] if g in gene_idx_map
            else 0.0 for g in union_genes]
    hm_norm = hm_data.div(hm_data.max(axis=1)+1e-10, axis=0)
    fig, ax = plt.subplots(
        figsize=(max(6, len(analyse_comps)*1.5),
                 max(8, len(union_genes)*0.28)))
    sns.heatmap(hm_norm, cmap='YlOrRd', ax=ax,
                linewidths=0.2,
                cbar_kws={'label':'Normalised loading'},
                xticklabels=True, yticklabels=True)
    ax.set_title(
        f'NMF loading heatmap — significant genes\n'
        f'C1: loading>mean+{LOADING_SD}SD  '
        f'C2: raw GAP t-test FDR q<{GENE_FDR_ALPHA}  '
        f'(independent criteria)',
        fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, 'nmf_loading_heatmap.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: nmf_loading_heatmap.png")

# ============================================================
# STEP 6 — RESOLVE SYMBOLS
# ============================================================
print("\nSTEP 6 — Resolving gene symbols")
all_resolved  = {}
all_named_dfs = {}

for _, row in analyse_comps.iterrows():
    comp_name = row['component']
    gdf       = all_sig_genes.get(comp_name, pd.DataFrame())
    if len(gdf) == 0:
        print(f"  {comp_name}: no significant genes — skipped")
        continue
    print(f"\n  {comp_name}  ({len(gdf):,} genes)...")
    resolved_syms, gene_map = resolve_symbols(
        gdf['gene'].tolist())
    time.sleep(0.3)

    named_rows = []
    for _, g_row in gdf.iterrows():
        gid = g_row['gene']
        if gid in gene_map:
            sym  = gene_map[gid]['symbol']
            name = gene_map[gid]['name']
        elif not gid.startswith('GENE_'):
            sym = gid; name = ''
        else:
            sym = gid; name = ''
        named_rows.append({
            'gene_id':   gid,
            'symbol':    sym,
            'gene_name': name,
            'loading':   g_row['loading'],
            'cohens_d':  g_row['cohens_d'],
            'p_value':   g_row['p_value'],
            'component': comp_name,
            'direction': row['direction'],
        })
    named_df = pd.DataFrame(named_rows)
    named_df.to_csv(
        os.path.join(OUT_DIR,
                     f"sig_genes_{comp_name}_named.csv"),
        index=False)
    print(f"  Saved: sig_genes_{comp_name}_named.csv")
    print(f"  Top 10:")
    print(named_df.head(10)[
        ['symbol','gene_name','loading','cohens_d']
    ].to_string(index=False))
    resolved_only = [
        r['symbol'] for r in named_rows
        if not r['symbol'].startswith('GENE_')]
    all_resolved[comp_name]  = resolved_only
    all_named_dfs[comp_name] = named_df
    print(f"  Resolved: {len(resolved_only):,} for GSEA/PPI")

# ============================================================
# STEP 7 — GSEA + ALL GO PLOTS PER COMPONENT
# ============================================================
print("\nSTEP 7 — GSEA + GO plots per component")

all_enr_results  = []
comp_enr_summary = {}

for _, row in analyse_comps.iterrows():
    comp_name    = row['component']
    direction    = row['direction']
    cohens_d     = row['cohens_d']
    top_resolved = all_resolved.get(comp_name, [])
    n_sig        = len(all_sig_genes.get(
        comp_name, pd.DataFrame()))

    print(f"\n  {comp_name}  ({direction}  d={cohens_d:.3f})"
          f"  {len(top_resolved):,} genes")
    if len(top_resolved) < 3:
        print(f"  Skipped — too few resolved genes")
        continue

    comp_dir = os.path.join(OUT_DIR, f'gsea_{comp_name}')
    os.makedirs(comp_dir, exist_ok=True)
    comp_enr_summary[comp_name] = {}
    col = RED if direction == 'AD' else BLUE

    for lib in GO_LIBS:
        sig = run_gsea(top_resolved, lib,
                       comp_name, direction)
        if sig is None:
            continue
        all_enr_results.append(sig)
        comp_enr_summary[comp_name][lib] = sig
        safe_lib = lib.replace('/','_').replace(' ','_')
        sig.to_csv(
            os.path.join(comp_dir,
                         f'gsea_{comp_name}_{safe_lib}.csv'),
            index=False)
        # Plot A: dot plot per library
        plot_dotplot(
            sig,
            title=(f'{comp_name}  ({direction})  '
                   f'd={cohens_d:.3f}\n'
                   f'{lib}  —  {len(top_resolved):,} genes\n'
                   f'(C1: loading>mean+{LOADING_SD}SD  '
                   f'C2: raw GAP t-test FDR)'),
            outpath=os.path.join(
                comp_dir,
                f'dot_{comp_name}_{safe_lib}.png'))

    libs_done = comp_enr_summary.get(comp_name, {})
    if not libs_done:
        continue

    # Plot B: bar chart
    plot_bar_chart(
        libs_done, comp_name, direction, cohens_d,
        outpath=os.path.join(
            comp_dir, f'bar_{comp_name}_all_libs.png'))
    print(f"    Saved: bar_{comp_name}_all_libs.png")

    # Plot C: bubble
    plot_bubble_summary(
        libs_done, comp_name, direction, cohens_d,
        outpath=os.path.join(
            comp_dir, f'bubble_{comp_name}.png'))
    print(f"    Saved: bubble_{comp_name}.png")

    # Plot D: heatmap
    plot_enrichment_heatmap(
        libs_done, comp_name, direction, cohens_d,
        outpath=os.path.join(
            comp_dir,
            f'heatmap_enrichment_{comp_name}.png'))
    print(f"    Saved: heatmap_enrichment_{comp_name}.png")

    # Plot E: combined multi-panel summary
    n_panels = len(libs_done)
    fig, axes_s = plt.subplots(
        1, n_panels,
        figsize=(min(10, 8*n_panels), 7),
        squeeze=False)
    axes_s = axes_s[0]
    for ax_i, lib in enumerate(libs_done):
        sp   = libs_done[lib].sort_values(
            'gene_ratio').head(10)
        ax   = axes_s[ax_i]
        nmin = sp['n_overlap'].min()
        nmax = sp['n_overlap'].max()
        sz   = 40+140*((sp['n_overlap']-nmin)/
                        (nmax-nmin+1))
        lc   = LIB_COLORS.get(lib, '#888888')
        sc   = ax.scatter(
            sp['gene_ratio'], range(len(sp)),
            c=sp['neg_log10_p'], s=sz, cmap='Reds',
            vmin=0, vmax=sp['neg_log10_p'].max()+0.5,
            edgecolors='grey', linewidths=0.3, zorder=3)
        ax.set_yticks(range(len(sp)))
        ax.set_yticklabels(sp['term'], fontsize=7)
        ax.set_xlabel('Gene ratio', fontsize=9)
        short = LIB_SHORT.get(lib, lib)
        ax.set_title(short, fontsize=9,
                     fontweight='bold', color=lc)
        ax.grid(axis='x', color='lightgrey',
                linewidth=0.4, zorder=0)
        plt.colorbar(sc, ax=ax, shrink=0.5,
                     label='-log10(p)').ax.tick_params(
                         labelsize=7)
    fig.suptitle(
        f'{comp_name}  ({direction})  d={cohens_d:.3f}\n'
        f'GSEA — {len(top_resolved):,} genes  '
        f'(C1: loading>mean+{LOADING_SD}SD  '
        f'C2: raw GAP FDR q<{GENE_FDR_ALPHA})',
        fontsize=10, color=col, y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(comp_dir,
                     f'gsea_summary_{comp_name}.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: gsea_summary_{comp_name}.png")

if all_enr_results:
    combined = pd.concat(all_enr_results, ignore_index=True)
    combined.to_csv(
        os.path.join(OUT_DIR, 'gsea_all_results.csv'),
        index=False)
    keep = [c for c in
            ['component','direction','library','term',
             'n_overlap','gene_ratio','p_value',
             'p_adj','genes']
            if c in combined.columns]
    (combined.groupby(['component','library'])
             .head(5)[keep]
             .to_csv(os.path.join(
                 OUT_DIR, 'gsea_manuscript_table.csv'),
                 index=False))
    print(f"\n  Saved: gsea_all_results.csv  "
          f"gsea_manuscript_table.csv")
else:
    print("\n  No GSEA results.")
    print("  Try lowering LOADING_SD to 2.0 or "
          "GENE_FDR_ALPHA to 0.10")

# ============================================================
# STEP 8A — PPI PER COMPONENT (all identified genes, no cap)
# ============================================================
print("\nSTEP 8A — PPI per significant component (all genes)")

per_comp_hubs = {}

for _, row in analyse_comps.iterrows():
    comp_name = row['component']
    direction = row['direction']
    cohens_d  = row['cohens_d']
    ppi_genes = all_resolved.get(comp_name, [])
    n_sig     = len(all_sig_genes.get(
        comp_name, pd.DataFrame()))
    comp_dir  = os.path.join(OUT_DIR, f'gsea_{comp_name}')
    os.makedirs(comp_dir, exist_ok=True)

    if len(ppi_genes) < 3:
        continue

    edges = query_string_all(ppi_genes, label=comp_name)
    time.sleep(2)

    hub_df = build_and_save_ppi(
        edges, ppi_genes,
        title_str=(
            f'PPI — {comp_name}  ({direction}  '
            f"d={cohens_d:.3f})\n"
            f'{n_sig:,} significant genes  →  '
            f'{len(ppi_genes):,} resolved  (all included)\n'
            f'C1: loading>mean+{LOADING_SD}SD  '
            f'C2: raw GAP FDR\n'
            f'STRING >= {STRING_SCORE}  |  '
            f'Orange = hub (degree >= {HUB_MIN_DEGREE})'),
        png_path=os.path.join(
            comp_dir, f'ppi_network_{comp_name}.png'),
        csv_path=os.path.join(
            comp_dir, f'ppi_hub_genes_{comp_name}.csv'),
    )
    if hub_df is not None:
        hub_df['component'] = comp_name
        hub_df['direction'] = direction
        per_comp_hubs[comp_name] = hub_df

if per_comp_hubs:
    all_hubs = pd.concat(per_comp_hubs.values(),
                         ignore_index=True)
    all_hubs.to_csv(
        os.path.join(OUT_DIR,
                     'ppi_hub_genes_per_component.csv'),
        index=False)
    print(f"\n  Saved: ppi_hub_genes_per_component.csv")

# ============================================================
# STEP 8B — COMBINED PPI (union of all components)
# ============================================================
print("\nSTEP 8B — Combined PPI across all components")

union_symbols = []
gene_comp_map = {}
for comp_name, resolved in all_resolved.items():
    for sym in resolved:
        if sym not in gene_comp_map:
            gene_comp_map[sym] = []
            union_symbols.append(sym)
        gene_comp_map[sym].append(comp_name)
union_symbols = list(dict.fromkeys(union_symbols))
print(f"  Union: {len(union_symbols):,} unique symbols")

comp_colors_list = [
    '#1D9E75','#534AB7','#D85A30','#185FA5',
    '#E8A838','#9B59B6','#2ECC71','#E74C3C',
]
comp_color_map = {
    row['component']: comp_colors_list[
        i % len(comp_colors_list)]
    for i, (_, row) in enumerate(analyse_comps.iterrows())
}
node_colors_map = {
    sym: comp_color_map.get(
        gene_comp_map[sym][0], '#AAAAAA')
    for sym in union_symbols
}

edges_combined = query_string_all(
    union_symbols, label='combined all components')
time.sleep(2)

comp_str = '  |  '.join(
    [f"{row['component']} ({row['direction']}  "
     f"d={row['cohens_d']:.2f})"
     for _, row in analyse_comps.iterrows()])

build_and_save_ppi(
    edges_combined, union_symbols,
    title_str=(
        f'Combined PPI — ALL significant NMF components\n'
        f'{comp_str}\n'
        f'{len(union_symbols):,} genes (all included, no cap)\n'
        f'C1: loading>mean+{LOADING_SD}SD  '
        f'C2: raw GAP FDR q<{GENE_FDR_ALPHA}\n'
        f'STRING >= {STRING_SCORE}  |  '
        f'Orange = hub  |  colour = component'),
    png_path=os.path.join(
        OUT_DIR,
        'ppi_network_COMBINED_all_components.png'),
    csv_path=os.path.join(
        OUT_DIR, 'ppi_hub_genes_COMBINED.csv'),
    node_colors_map=node_colors_map,
)

comp_map_df = pd.DataFrame([
    {'symbol':       sym,
     'components':   ','.join(gene_comp_map[sym]),
     'n_components': len(gene_comp_map[sym])}
    for sym in union_symbols
]).sort_values('n_components', ascending=False)
comp_map_df.to_csv(
    os.path.join(OUT_DIR,
                 'ppi_gene_component_membership.csv'),
    index=False)
print(f"  Saved: ppi_gene_component_membership.csv")

shared = comp_map_df[comp_map_df['n_components'] > 1]
if len(shared):
    print(f"\n  Genes shared across components "
          f"({len(shared):,}):")
    print(shared.head(15).to_string(index=False))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  NMF k              : {N_COMPONENTS}")
print(f"  Significant comps  : {len(sig_comps)}")
print(f"  Gene selection method:")
print(f"    C1: H loading > mean + {LOADING_SD}*SD")
print(f"        (component-specific genes)")
print(f"    C2: Welch t-test on RAW unweighted GAP scores")
print(f"        BH-FDR q < {GENE_FDR_ALPHA} within pool")
print(f"        (criteria are INDEPENDENT)")
print(f"    Cap: top {MAX_SIG_GENES} by |Cohen's d| if exceeded")
print(f"  PPI: ALL identified genes, no cap, STRING limit=0")
print()
for _, row in comp_df[comp_df['significant']].iterrows():
    comp  = row['component']
    n_sig = len(all_sig_genes.get(comp, pd.DataFrame()))
    n_res = len(all_resolved.get(comp, []))
    print(f"  {comp}  d={row['cohens_d']:.3f}  "
          f"q={row['q_value']:.4f}  "
          f"dir={row['direction']}  "
          f"sig_genes={n_sig:,}  "
          f"resolved={n_res:,}")
print(f"\n  Outputs → {OUT_DIR}")
print(f"    nmf_component_stats.csv")
print(f"    nmf_component_scores.png")
print(f"    nmf_component_volcano.png")
print(f"    nmf_loading_heatmap.png")
print(f"    gsea_all_results.csv")
print(f"    gsea_manuscript_table.csv")
print(f"    ppi_hub_genes_per_component.csv")
print(f"    ppi_gene_component_membership.csv")
print(f"    ppi_network_COMBINED_all_components.png")
print(f"    gsea_NMF_X/")
print(f"      sig_genes_NMF_X.csv + _named.csv")
print(f"      dot_NMF_X_<lib>.png")
print(f"      bar_NMF_X_all_libs.png")
print(f"      bubble_NMF_X.png")
print(f"      heatmap_enrichment_NMF_X.png")
print(f"      gsea_summary_NMF_X.png")
print(f"      ppi_network_NMF_X.png")
print(f"      ppi_hub_genes_NMF_X.csv")
print("="*60)