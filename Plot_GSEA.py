# ============================================================
# COMPLETE PIPELINE — GAP MATRIX → SIGNIFICANT GENES → ENRICHMENT → PPI
# No pre-saved CSVs needed. Runs end to end from gap_matrix.csv
#
# Steps:
#   1.  Load gap matrix
#   2.  Row-normalise
#   3.  Welch t-test per gene
#   4.  BH-FDR correction
#   5.  Save significant gene lists
#   6.  Volcano plot (log2FC x-axis, Cohen's d colourbar)
#   7.  Resolve GENE_<id> placeholders → real HGNC symbols
#   8.  GO + KEGG + Reactome enrichment (gseapy, correct background)
#   9.  Redundancy removal (Jaccard)
#   10. Publication dot plots (gene ratio + p-value + count)
#   11. PPI network via STRING API
#   12. Hub gene identification
# ============================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import gseapy as gp
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
GAP_DIR  = r'C:\Users\StujenskeLab\Documents\NAN151_workspace\Alzheimer\final_NMI_results\spearman'
GAP_CSV  = os.path.join(GAP_DIR, 'gap_matrix.csv')
OUT_DIR  = os.path.join(GAP_DIR, 'enrichment_publication')
os.makedirs(OUT_DIR, exist_ok=True)

FDR_ALPHA      = 0.05
HUB_MIN_DEGREE = 5
STRING_SCORE   = 400    # 400=medium, 700=high confidence
TOP_N_TERMS    = 15     # terms per dot plot

GO_LIBS = [
    'GO_Biological_Process_2023',
    'GO_Cellular_Component_2023',
    'GO_Molecular_Function_2023',
]
PATHWAY_LIBS = [
    'KEGG_2021_Human',
    'Reactome_2022',
    'WikiPathway_2023_Human',
]

# ============================================================
# STEP 1 — LOAD
# ============================================================
print("=" * 60)
print("STEP 1 — Loading gap matrix")
print("=" * 60)

df = pd.read_csv(GAP_CSV)
if 'subject' in df.columns:
    df = df.rename(columns={'subject': 'subject_id'})

meta_cols = [c for c in df.columns if c in ('subject_id', 'group')]
gene_cols = [c for c in df.columns if c not in meta_cols]

n_ad = sum(df['group'] == 'AD')
n_cn = sum(df['group'] == 'CN')
print(f"  Subjects : {len(df)}  (AD={n_ad}, CN={n_cn})")
print(f"  Genes    : {len(gene_cols):,}")

# ============================================================
# STEP 2 — ROW-NORMALISE
# Each subject's GAP vector divided by its own mean.
# Removes systematic AD > CN NMI bias caused by higher
# GMV variance in AD subjects.
# ============================================================
print("\nSTEP 2 — Row-normalising GAP scores")

gene_data      = df[gene_cols].copy()
row_means      = gene_data.mean(axis=1)
gene_data_norm = gene_data.div(row_means, axis=0)
df[gene_cols]  = gene_data_norm

ad_idx = df[df['group'] == 'AD'].index
cn_idx = df[df['group'] == 'CN'].index

# ============================================================
# STEP 3 — WELCH T-TEST PER GENE
# ============================================================
print("\nSTEP 3 — Welch t-test per gene")

results = []
for gene in tqdm(gene_cols, desc='  t-test'):
    ad_vals = gene_data_norm.loc[ad_idx, gene].dropna().values
    cn_vals = gene_data_norm.loc[cn_idx, gene].dropna().values

    if len(ad_vals) < 3 or len(cn_vals) < 3:
        continue

    t_stat, p_val = stats.ttest_ind(ad_vals, cn_vals, equal_var=False)
    effect        = ad_vals.mean() - cn_vals.mean()
    pooled_sd     = np.sqrt(
        (ad_vals.std(ddof=1)**2 + cn_vals.std(ddof=1)**2) / 2
    )
    d = effect / pooled_sd if pooled_sd > 0 else 0.0

    # log2 fold change — safe floor to avoid log(negative)
    ad_mean     = ad_vals.mean()
    cn_mean     = cn_vals.mean()
    ad_mean_pos = max(ad_mean, 1e-10)
    cn_mean_pos = max(cn_mean, 1e-10)
    lfc         = np.log2(ad_mean_pos / cn_mean_pos)

    results.append({
        'gene':      gene,
        'AD_mean':   ad_mean,
        'CN_mean':   cn_mean,
        'effect':    effect,
        'log2FC':    lfc,
        'cohens_d':  d,
        't_stat':    t_stat,
        'p_value':   p_val,
        'direction': 'AD' if effect > 0 else 'CN',
    })

stats_df = pd.DataFrame(results)
print(f"  Genes tested: {len(stats_df):,}")

# ============================================================
# STEP 4 — BH-FDR CORRECTION
# ============================================================
print("\nSTEP 4 — BH-FDR correction")

reject, q_vals, _, _ = multipletests(
    stats_df['p_value'].values, alpha=FDR_ALPHA, method='fdr_bh'
)
stats_df['q_value']     = q_vals
stats_df['significant'] = reject
stats_df = stats_df.sort_values('p_value').reset_index(drop=True)

sig    = stats_df[stats_df['significant']]
sig_ad = sig[sig['direction'] == 'AD'].sort_values(
    'cohens_d', ascending=False)
sig_cn = sig[sig['direction'] == 'CN'].sort_values('cohens_d')

print(f"  Total significant : {len(sig):,}")
print(f"    Higher in AD    : {len(sig_ad):,}")
print(f"    Higher in CN    : {len(sig_cn):,}")

print(f"\n  Top 10 AD genes:")
print(sig_ad[['gene','log2FC','cohens_d',
              'p_value','q_value']].head(10).to_string(index=False))
print(f"\n  Top 10 CN genes:")
print(sig_cn[['gene','log2FC','cohens_d',
              'p_value','q_value']].head(10).to_string(index=False))

# ============================================================
# STEP 5 — SAVE GENE LISTS
# ============================================================
stats_df.to_csv(os.path.join(OUT_DIR, 'ttest_fdr_results.csv'),
                index=False)
sig_ad.to_csv(os.path.join(OUT_DIR, 'sig_genes_AD.csv'), index=False)
sig_cn.to_csv(os.path.join(OUT_DIR, 'sig_genes_CN.csv'), index=False)
print(f"\n  Saved: ttest_fdr_results.csv, sig_genes_AD.csv, sig_genes_CN.csv")

ad_genes   = sig_ad['gene'].tolist()
cn_genes   = sig_cn['gene'].tolist()
background = stats_df['gene'].tolist()

# ============================================================
# STEP 6 — VOLCANO PLOT
# x-axis : log2 fold change (AD / CN)
# y-axis : −log10(p-value)
# colour : Cohen's d  (red=positive/AD, blue=negative/CN)
# ============================================================
print("\nSTEP 6 — Volcano plot")

fig, ax = plt.subplots(figsize=(12, 8))
nlp     = -np.log10(stats_df['p_value'].clip(lower=1e-300))
ns      = ~stats_df['significant']
ad_m    =  stats_df['significant'] & (stats_df['direction'] == 'AD')
cn_m    =  stats_df['significant'] & (stats_df['direction'] == 'CN')

max_d  = stats_df['cohens_d'].abs().max()
norm_d = plt.Normalize(vmin=-max_d, vmax=max_d)

ax.scatter(stats_df.loc[ns, 'log2FC'], nlp[ns],
           s=6, color='#CCCCCC', alpha=0.35, zorder=1,
           label=f'Not significant  (n={ns.sum():,})')

ax.scatter(
    stats_df.loc[ad_m, 'log2FC'], nlp[ad_m],
    c=stats_df.loc[ad_m, 'cohens_d'],
    cmap='RdYlBu_r', norm=norm_d,
    s=45, alpha=0.85, zorder=3,
    edgecolors='white', linewidths=0.3,
    label=f'Higher in AD  (n={ad_m.sum():,})',
)
sc = ax.scatter(
    stats_df.loc[cn_m, 'log2FC'], nlp[cn_m],
    c=stats_df.loc[cn_m, 'cohens_d'],
    cmap='RdYlBu_r', norm=norm_d,
    s=45, alpha=0.85, zorder=3,
    edgecolors='white', linewidths=0.3,
    label=f'Higher in CN  (n={cn_m.sum():,})',
)

cb = plt.colorbar(sc, ax=ax)
cb.set_label("Cohen's d  (effect size)", fontsize=10)

if len(sig):
    fdr_p = sig['p_value'].max()
    ax.axhline(
        -np.log10(fdr_p), color='black',
        linestyle='--', linewidth=1,
        label=f'BH q = {FDR_ALPHA}  (p ≈ {fdr_p:.2e})',
    )

ax.axvline(0, color='black', linewidth=0.7)

if len(sig):
    top_lbl = sig.reindex(
        sig['log2FC'].abs().sort_values(ascending=False).index
    ).head(20)
    med_nlp = nlp[stats_df['significant']].median()
    for _, row in top_lbl.iterrows():
        gp_val = -np.log10(max(row['p_value'], 1e-300))
        col    = '#D85A30' if row['direction'] == 'AD' else '#185FA5'
        off    = 9 if gp_val >= med_nlp else -12
        ax.annotate(
            row['gene'],
            xy=(row['log2FC'], gp_val),
            xytext=(0, off),
            textcoords='offset points',
            fontsize=7.5, ha='center',
            color=col, fontweight='bold',
        )

ax.set_xlabel(
    'log₂ fold change  (AD / CN normalised GAP score)\n'
    'positive → higher in AD   |   negative → higher in CN',
    fontsize=11)
ax.set_ylabel('−log₁₀(p-value)', fontsize=11)
ax.set_title(
    f"Volcano plot — colored by Cohen's d effect size\n"
    f"{len(stats_df):,} genes tested  |  "
    f"{len(sig):,} significant (BH q<{FDR_ALPHA})",
    fontsize=11)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'volcano_plot.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: volcano_plot.png")

# ============================================================
# STEP 7 — RESOLVE GENE_<id> → REAL HGNC SYMBOLS
# STRING and Enrichr both require official gene symbols.
# GENE_114770 → e.g. APOE via mygene.info API (free, no key).
# ============================================================
print("\nSTEP 7 — Resolving gene symbols via mygene.info")


def resolve_gene_symbols(gene_list, batch_size=1000):
    """
    Convert GENE_<entrezid> placeholders → real HGNC symbols.
    Genes already named with real symbols pass through unchanged.
    Genes with no match are dropped (enrichment tools cannot use
    placeholder names).
    Returns (resolved_symbol_list, placeholder_to_symbol_dict).
    """
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

    print(f"  Querying {len(placeholders):,} Entrez IDs "
          f"({len(real_genes)} already named)...")

    url     = 'https://mygene.info/v3/gene'
    id_list = list(placeholders.values())
    sym_map = {}   # entrez_id (int) -> symbol (str)

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        try:
            resp = requests.post(
                url,
                json={
                    'ids':     [str(x) for x in batch],
                    'fields':  'symbol',
                    'species': 'human',
                },
                timeout=30,
            )
            resp.raise_for_status()
            for rec in resp.json():
                eid = rec.get('_id')
                sym = rec.get('symbol')
                if eid and sym:
                    try:
                        sym_map[int(eid)] = str(sym)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            print(f"  [mygene error batch {i}] {e}")
        time.sleep(0.3)

    gene_to_symbol = {}
    resolved       = list(real_genes)
    for placeholder, eid in placeholders.items():
        sym = sym_map.get(eid)
        if sym:
            resolved.append(sym)
            gene_to_symbol[placeholder] = sym
        # genes with no match are dropped — enrichment tools
        # cannot use GENE_<id> format

    found    = len(gene_to_symbol)
    dropped  = len(placeholders) - found
    print(f"  Resolved: {found:,} / {len(placeholders):,}  "
          f"({dropped:,} not found — dropped)")
    return resolved, gene_to_symbol


# Resolve all three gene lists
print("\n  AD-up genes...")
ad_resolved, ad_sym_map = resolve_gene_symbols(ad_genes)
time.sleep(0.5)

print("\n  CN-up genes...")
cn_resolved, cn_sym_map = resolve_gene_symbols(cn_genes)
time.sleep(0.5)

print("\n  Background genes (all tested)...")
bg_resolved, _          = resolve_gene_symbols(background)

print(f"\n  AD-up resolved   : {len(ad_resolved):,}")
print(f"  CN-up resolved   : {len(cn_resolved):,}")
print(f"  Background       : {len(bg_resolved):,}")

# Save resolved symbol lists for reference / supplementary table
pd.DataFrame({'gene_placeholder': list(ad_sym_map.keys()),
              'symbol':           list(ad_sym_map.values())}
             ).to_csv(os.path.join(OUT_DIR,
                                   'ad_genes_symbols.csv'), index=False)
pd.DataFrame({'gene_placeholder': list(cn_sym_map.keys()),
              'symbol':           list(cn_sym_map.values())}
             ).to_csv(os.path.join(OUT_DIR,
                                   'cn_genes_symbols.csv'), index=False)

# ============================================================
# ENRICHMENT HELPERS
# ============================================================

def run_enr(gene_list, background_list, library, group_label,
            p_cutoff=0.05, min_genes=3):
    """
    Run ORA via gseapy Enrichr with correct background.
    Handles column name differences across gseapy versions.
    """
    if len(gene_list) < 5:
        return None
    try:
        enr = gp.enrichr(
            gene_list  = gene_list,
            gene_sets  = library,
            background = background_list,
            outdir     = None,
            verbose    = False,
        )
    except Exception as e:
        print(f"      [enrichr error] {e}")
        return None

    res = enr.results.copy()
    if res is None or len(res) == 0:
        return None

    # Normalise column names across gseapy versions
    col_map = {}
    for col in res.columns:
        cl = (col.lower()
              .replace(' ', '_')
              .replace('-', '_')
              .replace('.', '_'))
        if cl == 'term':
            col_map[col] = 'term'
        elif cl == 'overlap':
            col_map[col] = 'overlap'
        elif cl in ('p_value', 'pvalue', 'p_val'):
            col_map[col] = 'p_value'
        elif cl in ('adjusted_p_value', 'adj__p_value',
                    'fdr', 'padj', 'adj_p_val',
                    'adjusted_p_value'):
            col_map[col] = 'p_adj'
        elif cl in ('genes', 'gene_list', 'lead_genes', 'gene'):
            col_map[col] = 'genes'
        elif cl in ('combined_score', 'combined_score', 'score'):
            col_map[col] = 'combined_score'
    res = res.rename(columns=col_map)

    # Parse overlap into n_overlap and n_list
    if 'overlap' in res.columns:
        def safe_split(x, idx):
            try:
                return int(str(x).split('/')[idx])
            except Exception:
                return 0
        res['n_overlap'] = res['overlap'].apply(
            lambda x: safe_split(x, 0))
        res['n_list']    = res['overlap'].apply(
            lambda x: safe_split(x, 1))
    else:
        found_overlap = False
        for col in res.columns:
            sample = str(res[col].iloc[0]) if len(res) else ''
            if '/' in sample:
                res['n_overlap'] = res[col].apply(
                    lambda x: int(str(x).split('/')[0]))
                res['n_list']    = res[col].apply(
                    lambda x: int(str(x).split('/')[1]))
                found_overlap = True
                break
        if not found_overlap:
            res['n_overlap'] = 0
            res['n_list']    = len(gene_list)

    # Ensure p_adj exists
    if 'p_adj' not in res.columns:
        if 'p_value' in res.columns:
            _, padj, _, _ = multipletests(
                res['p_value'].fillna(1).values, method='fdr_bh')
            res['p_adj'] = padj
        else:
            res['p_adj'] = 1.0

    # Ensure term column exists
    if 'term' not in res.columns:
        for col in res.columns:
            if (res[col].dtype == object and
                    col not in ('genes', 'overlap',
                                'p_adj', 'p_value')):
                res = res.rename(columns={col: 'term'})
                break

    res['gene_ratio']  = res['n_overlap'] / (res['n_list'] + 1e-10)
    res['neg_log10_p'] = -np.log10(res['p_adj'].clip(lower=1e-300))
    res['library']     = library
    res['group']       = group_label

    res = res[
        (res['p_adj']     < p_cutoff) &
        (res['n_overlap'] >= min_genes)
    ].copy()

    return res.sort_values('p_adj').reset_index(drop=True)


def remove_redundant(df, threshold=0.5):
    """
    Remove redundant GO terms using Jaccard similarity.
    Keeps the most significant term when two terms share
    >= threshold fraction of their gene members.
    """
    if df is None or len(df) == 0:
        return df
    gene_sets = {}
    for _, row in df.iterrows():
        gene_sets[row['term']] = set(
            str(row.get('genes', '')).split(';'))
    terms = df['term'].tolist()
    keep  = [True] * len(terms)
    for i in range(len(terms)):
        if not keep[i]:
            continue
        for j in range(i+1, len(terms)):
            if not keep[j]:
                continue
            a = gene_sets.get(terms[i], set())
            b = gene_sets.get(terms[j], set())
            u = a | b
            if u and len(a & b) / len(u) >= threshold:
                keep[j] = False
    return df[
        [t for t, k in zip(df.index, keep) if k]
    ].reset_index(drop=True)


def pub_dotplot(df, title, outpath, top_n=TOP_N_TERMS):
    """
    Publication dot plot:
      x     = gene ratio (overlap / list size)
      y     = GO / pathway term
      size  = number of genes overlapping
      colour= −log10(adjusted p-value)
    """
    if df is None or len(df) == 0:
        return
    plot_df = df.head(top_n).sort_values('gene_ratio')

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(plot_df) * 0.45 + 2)))

    n_min = plot_df['n_overlap'].min()
    n_max = plot_df['n_overlap'].max()
    sizes = 40 + 200 * (
        (plot_df['n_overlap'] - n_min) / (n_max - n_min + 1)
    )

    sc = ax.scatter(
        plot_df['gene_ratio'],
        range(len(plot_df)),
        c=plot_df['neg_log10_p'],
        s=sizes,
        cmap='RdYlBu_r',
        vmin=plot_df['neg_log10_p'].min(),
        vmax=plot_df['neg_log10_p'].max(),
        edgecolors='grey', linewidths=0.4, zorder=3,
    )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['term'], fontsize=9)
    ax.set_xlabel(
        'Gene ratio  (genes in term / genes in list)',
        fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(axis='x', color='lightgrey', linewidth=0.5, zorder=0)

    cb = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02)
    cb.set_label('−log10(adjusted p-value)', fontsize=9)

    for s, lbl in [
        (40,  'low overlap'),
        (140, 'medium'),
        (240, 'high overlap'),
    ]:
        ax.scatter([], [], s=s, c='grey', alpha=0.6,
                   label=lbl, edgecolors='grey', linewidths=0.4)
    ax.legend(title='Gene count', fontsize=8,
              title_fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {os.path.basename(outpath)}")


# ============================================================
# STEP 8 — GO + PATHWAY ENRICHMENT (with real symbols)
# ============================================================
print("\nSTEP 8 — GO + Pathway enrichment")

all_enr = []

for libs, ltype in [
    (GO_LIBS,      'GO'),
    (PATHWAY_LIBS, 'Pathway'),
]:
    print(f"\n  {ltype}")
    for lib in libs:
        for genes, grp in [
            (ad_resolved, 'AD_up'),
            (cn_resolved, 'CN_up'),
        ]:
            print(f"\n    {lib} — {grp} ({len(genes):,} genes)")
            res = run_enr(genes, bg_resolved, lib, grp)
            res = remove_redundant(res, threshold=0.5)
            if res is not None and len(res):
                all_enr.append(res)
                print(f"      {len(res)} terms after redundancy filter")
                pub_dotplot(
                    res,
                    title=(
                        f'{lib}\n'
                        f'{grp}  (n={len(genes):,} genes  '
                        f'background={len(bg_resolved):,})'
                    ),
                    outpath=os.path.join(
                        OUT_DIR, f'dot_{lib}_{grp}.png'),
                )
            else:
                print(f"      No significant terms")

if all_enr:
    combined = pd.concat(all_enr, ignore_index=True)
    combined.to_csv(
        os.path.join(OUT_DIR, 'enrichment_all_results.csv'),
        index=False)
    keep_cols = [c for c in
                 ['group', 'library', 'term', 'n_overlap',
                  'gene_ratio', 'p_value', 'p_adj', 'genes']
                 if c in combined.columns]
    (combined
     .groupby(['library', 'group'])
     .head(5)[keep_cols]
     .to_csv(os.path.join(OUT_DIR,
                          'enrichment_manuscript_table.csv'),
             index=False))
    print(f"\n  Saved: enrichment_all_results.csv")
    print(f"  Saved: enrichment_manuscript_table.csv")

# ============================================================
# PPI HELPERS
# ============================================================

def get_string_ppi(gene_list, species=9606,
                   score_threshold=STRING_SCORE):
    """Query STRING API for protein-protein interactions."""
    genes = [g for g in gene_list if g][:500]
    try:
        resp = requests.post(
            'https://string-db.org/api/json/network',
            data={
                'identifiers':     '\r'.join(genes),
                'species':         species,
                'required_score':  score_threshold,
                'caller_identity': 'gap_score_analysis',
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            print(f"  [STRING] No interactions returned")
            return None
        return pd.DataFrame(data)
    except Exception as e:
        print(f"  [STRING error] {e}")
        return None


def build_ppi(edges_df, gene_list, group_label,
              hub_min_degree=HUB_MIN_DEGREE):
    """
    Build NetworkX PPI graph from STRING edges.
    Identifies hub genes by degree threshold.
    Saves network plot and hub gene CSV.
    """
    if edges_df is None or len(edges_df) == 0:
        print(f"  No PPI edges returned for {group_label}")
        return

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        a = row.get('preferredName_A', row.get('stringId_A', ''))
        b = row.get('preferredName_B', row.get('stringId_B', ''))
        s = float(row.get('score', 0))
        if a and b and a != b:
            G.add_edge(a, b, weight=s)

    # Keep only nodes that are in our gene list
    keep = set(gene_list) & set(G.nodes)
    G    = G.subgraph(keep).copy()
    degs = dict(G.degree())

    print(f"  {group_label}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    if G.number_of_nodes() == 0:
        print(f"  No overlapping nodes — check gene symbols match STRING")
        return

    # Hub genes — degree >= hub_min_degree
    hubs = sorted(
        [(n, d) for n, d in degs.items()
         if d >= hub_min_degree],
        key=lambda x: x[1], reverse=True,
    )
    hub_df = pd.DataFrame(hubs, columns=['gene', 'degree'])
    hub_df['group'] = group_label
    hub_df.to_csv(
        os.path.join(OUT_DIR, f'hub_genes_{group_label}.csv'),
        index=False)
    print(f"  Hub genes (degree >= {hub_min_degree}): {len(hubs)}")
    if hubs:
        print("  " + ", ".join(
            [f"{g}({d})" for g, d in hubs[:10]]))

    # Network plot
    fig, ax  = plt.subplots(figsize=(13, 11))
    pos      = nx.spring_layout(G, seed=42, k=0.5)
    hub_set  = {g for g, _ in hubs}
    n_sizes  = [30 + degs.get(n, 1) * 25 for n in G.nodes]
    n_colors = ['#D85A30' if n in hub_set
                else '#AAAAAA' for n in G.nodes]

    nx.draw_networkx_edges(
        G, pos, alpha=0.3, edge_color='grey', ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_size=n_sizes,
        node_color=n_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in G.nodes if n in hub_set},
        font_size=7, ax=ax)

    ax.set_title(
        f'PPI network — {group_label}\n'
        f'{G.number_of_nodes()} proteins  '
        f'{G.number_of_edges()} interactions\n'
        f'STRING confidence ≥ {STRING_SCORE}  |  '
        f'Orange = hub genes (degree ≥ {hub_min_degree})',
        fontsize=11)
    ax.axis('off')
    ax.legend(handles=[
        mpatches.Patch(color='#D85A30', label='Hub gene'),
        mpatches.Patch(color='#AAAAAA', label='Other'),
    ], fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'ppi_network_{group_label}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ppi_network_{group_label}.png")


# ============================================================
# STEP 9 — PPI NETWORK (with resolved symbols)
# ============================================================
print("\nSTEP 9 — PPI network (STRING)")

print("\n  Querying STRING for AD-up genes...")
edges_ad = get_string_ppi(ad_resolved, score_threshold=STRING_SCORE)
time.sleep(1)
build_ppi(edges_ad, ad_resolved, 'AD_up')

print("\n  Querying STRING for CN-up genes...")
edges_cn = get_string_ppi(cn_resolved, score_threshold=STRING_SCORE)
time.sleep(1)
build_ppi(edges_cn, cn_resolved, 'CN_up')

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ALL OUTPUTS")
print("=" * 60)
print(f"  Directory: {OUT_DIR}\n")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f:<55} {size/1024:>7.1f} KB")
print("=" * 60)