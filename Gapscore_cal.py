"""
gap_score_20genes.py
====================
Computes per-subject GAP scores (NMI) between each subject's gray matter
volume map and 20 specific gene expression maps (10 AD-related, 10 not related).

Output:
  - gap_matrix_20genes.csv      : (92 subjects x 20 genes) NMI scores
  - ttest_results_20genes.csv   : t-stat, p-value, Cohen's d per gene
  - boxplot_per_gene.png        : AD vs CN NMI distributions per gene
  - heatmap_gap_matrix.png      : subject x gene NMI heatmap

Pipeline:
  1. Load subject GMV maps (SimpleITK)
  2. Build brain mask + voxel-wise z-score
  3. Load each gene map (SimpleITK resample to GMV grid)
  4. For each subject x gene: NMI(subject GMV, gene map)
  5. Independent t-test per gene (AD vs CN) + FDR correction
  6. Plots

Requirements:
    pip install nibabel scipy statsmodels tqdm matplotlib seaborn SimpleITK
"""

import os
import zipfile
import tempfile
import warnings
import logging
import sys

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
#  CONFIG
# =============================================================================
BASE_DIR      = r'C:\Users\StujenskeLab\Documents\NAN151_workspace\Alzheimer'
Gene_DIR       = r'U:\Research\AZ'
CSV_PATH      = os.path.join(BASE_DIR, 'subjects_labels.csv')
GENE_MAPS_DIR = os.path.join(Gene_DIR, 'meduni_gene_maps')
OUTPUT_DIR    = os.path.join(BASE_DIR, 'gap_results_20genes')

BRAIN_MASK_THRESHOLD = 0.1   # mean GMV threshold for brain mask
NMI_BINS             = 32    # histogram bins for NMI computation

# 20 genes: Entrez ID 
# group: 'AD-related' or 'Not related'
GENE_LIST = {
    # AD-related (GWAS hits)
    348:    ('APOE',   'AD-related'),
    54209:  ('TREM2',  'AD-related'),
    10347:  ('ABCA7',  'AD-related'),
    6653:   ('SORL1',  'AD-related'),
    23607:  ('CD2AP',  'AD-related'),
    1191:   ('CLU',    'AD-related'),
    274:    ('BIN1',   'AD-related'),
    114815: ('SORCS1', 'AD-related'),
    123041: ('SLC24A4','AD-related'),
    2185:   ('PTK2B',  'AD-related'),
    # Not AD-related (negative controls)
    672:    ('BRCA1',   'Not related'),
    324:    ('APC',     'Not related'),
    1080:   ('CFTR',    'Not related'),
    3043:   ('HBB',     'Not related'),
    3625:   ('INHBB',   'Not related'),
    4625:   ('MYH7',    'Not related'),
    4080:   ('PAX6',    'Not related'),
    5172:   ('SLC26A4', 'Not related'),
    5251:   ('PHEX',    'Not related'),
    3861:   ('KRT14',   'Not related'),
}
# =============================================================================


# ── NMI ───────────────────────────────────────────────────────────────────────
def nmi_1d(a, b, n_bins=NMI_BINS):
    """
    Normalized Mutual Information between two 1D float arrays.
    NMI = 2 * MI(X,Y) / (H(X) + H(Y)),  range [0, 1].
    Returns np.nan if arrays are too short or degenerate.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 10:
        return np.nan

    hist_2d, _, _ = np.histogram2d(a, b, bins=n_bins)
    hist_2d = hist_2d + 1e-10          # Laplace smoothing

    p_xy = hist_2d / hist_2d.sum()
    p_x  = p_xy.sum(axis=1, keepdims=True)
    p_y  = p_xy.sum(axis=0, keepdims=True)

    mi  = (p_xy * np.log(p_xy / (p_x * p_y))).sum()
    h_x = -(p_x * np.log(p_x + 1e-10)).sum()
    h_y = -(p_y * np.log(p_y + 1e-10)).sum()

    denom = h_x + h_y
    return float(2.0 * mi / denom) if denom != 0 else np.nan


# ── Cohen's d ─────────────────────────────────────────────────────────────────
def cohens_d(g1, g2):
    """Pooled-SD Cohen's d between two 1D arrays."""
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 0 else 0.0


# ── SimpleITK helpers ─────────────────────────────────────────────────────────
def load_gmv_sitk(nii_path):
    """Load a GMV NIfTI with SimpleITK. Returns (sitk_image, float32 array x,y,z)."""
    img   = sitk.ReadImage(str(nii_path), sitk.sitkFloat32)
    arr   = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, arr.transpose(2, 1, 0)   # sitk z,y,x -> x,y,z


def resample_gene_to_gmv(gene_nii_path, ref_sitk_image):
    """
    Resample gene NIfTI to the exact GMV voxel grid using
    sitk.ResampleImageFilter with linear interpolation.
    Returns float32 array (x, y, z).
    """
    gene_img = sitk.ReadImage(str(gene_nii_path), sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampled = resampler.Execute(gene_img)
    arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
    return arr.transpose(2, 1, 0)


def apply_brain_mask_sitk(volume_array, mask_array):
    """
    Zero out non-brain voxels using sitk.MaskImageFilter.
    volume_array, mask_array: (x, y, z) numpy arrays.
    """
    vol_sitk  = sitk.GetImageFromArray(volume_array.transpose(2, 1, 0))
    mask_sitk = sitk.GetImageFromArray(mask_array.transpose(2, 1, 0).astype(np.uint8))
    f = sitk.MaskImageFilter()
    f.SetOutsideValue(0.0)
    masked = f.Execute(vol_sitk, mask_sitk)
    return sitk.GetArrayFromImage(masked).astype(np.float32).transpose(2, 1, 0)


def voxelwise_zscore(matrix):
    """
    Z-score a (N_subjects x N_voxels) matrix voxel-wise (across subjects).
    Flat voxels (std=0) are left unchanged.
    """
    mean = matrix.mean(axis=0)
    std  = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def load_gene_map(zip_path, ref_sitk_image, ref_shape, brain_mask):
    """
    Load a gene map from ZIP, resample to GMV grid via SimpleITK,
    apply brain mask, return 1D brain-voxel array or None on failure.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            nii_names = [
                n for n in z.namelist()
                if n.endswith('_mRNA.nii') and 'mirr' not in n
            ]
            if not nii_names:
                log.warning(f'  No _mRNA.nii found in {zip_path}')
                return None
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extract(nii_names[0], tmpdir)
                nii_path = os.path.join(tmpdir, nii_names[0])
                gene_arr    = resample_gene_to_gmv(nii_path, ref_sitk_image)
                mask_3d     = brain_mask.reshape(ref_shape).astype(np.uint8)
                gene_masked = apply_brain_mask_sitk(gene_arr, mask_3d)
                return gene_masked.flatten()[brain_mask]
    except Exception as e:
        log.warning(f'  Failed to load {zip_path}: {e}')
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_single_boxplot(gene_id, symbol, gene_group, ad_vals, cn_vals,
                        p_value, p_fdr, cohens_d_val, output_dir):
    """
    Save one boxplot for a single gene.
    Filename: boxplot_{gene_id}_{symbol}.png
    Shows CN (blue) vs AD (red) NMI distributions with jittered data points,
    t-test p-value, FDR q-value, and Cohen's d in the title.
    """
    fig, ax = plt.subplots(figsize=(5, 6))

    # Significance stars
    if p_fdr < 0.001:
        sig_star = '***'
    elif p_fdr < 0.01:
        sig_star = '**'
    elif p_fdr < 0.05:
        sig_star = '*'
    else:
        sig_star = 'ns'

    group_label = 'AD-related (GWAS)' if gene_group == 'AD-related' else 'Negative control'
    title_color = '#8B0000' if gene_group == 'AD-related' else '#1a1a2e'

    # Boxplots
    bp = ax.boxplot(
        [cn_vals, ad_vals],
        labels=['CN', 'AD'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.4),
    )
    bp['boxes'][0].set_facecolor('#6BAED6')   # CN = blue
    bp['boxes'][1].set_facecolor('#E07B7B')   # AD = red
    bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_alpha(0.75)

    # Jittered individual points
    np.random.seed(42)
    for j, (vals, color) in enumerate(
        zip([cn_vals, ad_vals], ['#2166AC', '#B2182B']), start=1
    ):
        ax.scatter(
            np.random.normal(j, 0.07, size=len(vals)),
            vals, alpha=0.6, s=22, color=color, zorder=3,
            edgecolors='white', linewidths=0.4
        )

    # Significance bracket between CN and AD boxes
    y_max   = max(np.nanmax(cn_vals), np.nanmax(ad_vals))
    y_range = y_max - min(np.nanmin(cn_vals), np.nanmin(ad_vals))
    y_bar   = y_max + y_range * 0.08
    ax.plot([1, 1, 2, 2], [y_bar - y_range*0.02, y_bar, y_bar, y_bar - y_range*0.02],
            color='black', linewidth=1.2)
    ax.text(1.5, y_bar + y_range * 0.01, sig_star,
            ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('NMI (GAP score)', fontsize=12)
    ax.set_xlabel('Group', fontsize=12)
    ax.tick_params(labelsize=11)

    ax.set_title(
        f'Gene ID: {gene_id}  |  {symbol}\n'
        f'{group_label}\n'
        f'p = {p_value:.4f}   q(FDR) = {p_fdr:.4f}   d = {cohens_d_val:.3f}   {sig_star}',
        fontsize=10, color=title_color, pad=8
    )

    plt.tight_layout()
    fname = f'boxplot_{gene_id}_{symbol}.png'
    out   = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_all_boxplots(gap_df, ttest_df, output_dir):
    """
    Save one boxplot PNG per gene, named boxplot_{gene_id}_{symbol}.png.
    Also saves a summary grid (all 20 genes on one figure) as
    boxplot_all_genes_grid.png.
    """
    boxplot_dir = os.path.join(output_dir, 'boxplots_per_gene')
    os.makedirs(boxplot_dir, exist_ok=True)

    ad_data = gap_df[gap_df['group'] == 'AD']
    cn_data = gap_df[gap_df['group'] == 'CN']

    saved = []
    for _, row in ttest_df.iterrows():
        sym     = row['gene_symbol']
        gid     = row['gene_id']
        ad_vals = ad_data[sym].dropna().values
        cn_vals = cn_data[sym].dropna().values

        path = plot_single_boxplot(
            gene_id     = gid,
            symbol      = sym,
            gene_group  = row['gene_group'],
            ad_vals     = ad_vals,
            cn_vals     = cn_vals,
            p_value     = row['p_value'],
            p_fdr       = row['p_fdr'],
            cohens_d_val= row['cohens_d'],
            output_dir  = boxplot_dir,
        )
        log.info(f'    Saved boxplot: {os.path.basename(path)}')
        saved.append(path)

    # ── Summary grid: all 20 on one figure ───────────────────────────────
    n_genes = len(ttest_df)
    ncols   = 5
    nrows   = int(np.ceil(n_genes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 5))
    axes = axes.flatten()

    np.random.seed(42)
    for i, (_, row) in enumerate(ttest_df.iterrows()):
        ax      = axes[i]
        sym     = row['gene_symbol']
        gid     = row['gene_id']
        ad_vals = ad_data[sym].dropna().values
        cn_vals = cn_data[sym].dropna().values

        if p_fdr := row['p_fdr'] < 0.001:
            sig_star = '***'
        elif row['p_fdr'] < 0.01:
            sig_star = '**'
        elif row['p_fdr'] < 0.05:
            sig_star = '*'
        else:
            sig_star = 'ns'

        bp = ax.boxplot(
            [cn_vals, ad_vals], labels=['CN', 'AD'],
            patch_artist=True, widths=0.45,
            medianprops=dict(color='black', linewidth=2),
        )
        bp['boxes'][0].set_facecolor('#6BAED6')
        bp['boxes'][1].set_facecolor('#E07B7B')
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_alpha(0.75)

        for j, (vals, color) in enumerate(
            zip([cn_vals, ad_vals], ['#2166AC', '#B2182B']), start=1
        ):
            ax.scatter(np.random.normal(j, 0.07, size=len(vals)),
                       vals, alpha=0.5, s=10, color=color, zorder=3)

        title_color = '#8B0000' if row['gene_group'] == 'AD-related' else '#1a1a2e'
        ax.set_title(
            f'ID:{gid}  {sym}\n'
            f'p={row["p_value"]:.3f}  q={row["p_fdr"]:.3f}\n'
            f'd={row["cohens_d"]:.2f}  {sig_star}',
            fontsize=8, color=title_color, pad=3
        )
        ax.set_ylabel('NMI', fontsize=7)
        ax.tick_params(labelsize=8)

    for j in range(n_genes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        'GAP Scores (NMI) per Gene — AD vs CN\n'
        'Red title = AD-related (GWAS)   Black title = Negative control\n'
        '* FDR<0.05   ** FDR<0.01   *** FDR<0.001   ns = not significant',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    grid_out = os.path.join(output_dir, 'boxplot_all_genes_grid.png')
    plt.savefig(grid_out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'    Saved summary grid: {grid_out}')


def plot_volcano(ttest_df, output_dir):
    """
    Volcano plot: x = Cohen's d (effect size, AD - CN direction)
                  y = -log10(raw p-value)
    Each dot is one gene, labeled by symbol and gene ID.
    AD-related genes = red markers, negative controls = blue markers.
    FDR threshold line drawn at q = 0.05.
    Significant genes (FDR < 0.05) are filled; non-significant are hollow.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Compute -log10 p
    ttest_df = ttest_df.copy()
    ttest_df['neg_log10_p'] = -np.log10(ttest_df['p_value'].clip(lower=1e-10))

    # FDR threshold line: find the raw p-value that corresponds to q=0.05
    # (draw a horizontal guide at the p of the last significant gene)
    sig_rows = ttest_df[ttest_df['fdr_significant']]
    if len(sig_rows) > 0:
        p_threshold = sig_rows['p_value'].max()
        ax.axhline(-np.log10(p_threshold), color='gray', linestyle='--',
                   linewidth=1.2, label=f'FDR=0.05 (p~{p_threshold:.3f})')

    # Plot each gene
    for _, row in ttest_df.iterrows():
        is_ad    = row['gene_group'] == 'AD-related'
        is_sig   = row['fdr_significant']
        color    = '#B2182B' if is_ad else '#2166AC'
        marker   = 'o'
        face     = color if is_sig else 'none'     # filled=significant
        edge     = color
        size     = 120 if is_sig else 70

        ax.scatter(
            row['cohens_d'], row['neg_log10_p'],
            s=size, marker=marker,
            facecolors=face, edgecolors=edge,
            linewidths=1.5, zorder=3
        )

        # Label: "SYMBOL\n(ID)"
        va_offset = 6 if row['neg_log10_p'] > ttest_df['neg_log10_p'].median() else -14
        ax.annotate(
            f"{row['gene_symbol']}\n({int(row['gene_id'])})",
            xy=(row['cohens_d'], row['neg_log10_p']),
            xytext=(0, va_offset), textcoords='offset points',
            fontsize=7.5, ha='center', va='bottom',
            color=color, fontweight='bold' if is_sig else 'normal'
        )

    # Reference lines
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#B2182B',
               markeredgecolor='#B2182B', markersize=9, label='AD-related (GWAS) — significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#B2182B', markersize=9, label='AD-related (GWAS) — not significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC',
               markeredgecolor='#2166AC', markersize=9, label='Negative control — significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#2166AC', markersize=9, label='Negative control — not significant'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left',
              framealpha=0.9)

    ax.set_xlabel("Cohen's d  (AD - CN GAP score difference)", fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title(
        'Volcano Plot — GAP Score Differences (NMI) between AD and CN\n'
        'x-axis: effect size (Cohen\'s d)   y-axis: statistical significance\n'
        'Filled markers = FDR significant   Gene ID shown in parentheses',
        fontsize=11
    )
    plt.tight_layout()
    out = os.path.join(output_dir, 'volcano_plot.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'    Saved: {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info('=' * 60)
    log.info('GAP Score Analysis — 20 Genes, Per-Subject NMI')
    log.info('=' * 60)
    log.info(f'SimpleITK version : {sitk.Version.VersionString()}')
    log.info(f'Genes             : {len(GENE_LIST)}')
    log.info(f'NMI bins          : {NMI_BINS}')
    log.info(f'Brain mask thresh : {BRAIN_MASK_THRESHOLD}')

    # ── [1] Load subject labels ───────────────────────────────────────────
    log.info('\n[1] Loading subject labels...')
    df = pd.read_csv(CSV_PATH)
    log.info(f'    Subjects : {len(df)}  (AD={( df["group"]=="AD").sum()}, CN={(df["group"]=="CN").sum()})')

    # ── [2] Load GMV maps via SimpleITK ──────────────────────────────────
    log.info('\n[2] Loading GMV maps (SimpleITK)...')
    ref_nib       = nib.load(df['gm_file'].iloc[0])
    ref_shape     = ref_nib.shape
    n_vox         = int(np.prod(ref_shape))
    gmv_matrix    = np.zeros((len(df), n_vox), dtype=np.float32)
    ref_sitk_image = None

    for i, row in tqdm(df.iterrows(), total=len(df), desc='    sitk.ReadImage'):
        try:
            sitk_img, arr = load_gmv_sitk(row['gm_file'])
            if ref_sitk_image is None:
                ref_sitk_image = sitk_img
            gmv_matrix[i] = arr.flatten()
        except Exception as e:
            log.warning(f'    Could not load {row["gm_file"]}: {e}')

    # Brain mask
    brain_mask = gmv_matrix.mean(axis=0) > BRAIN_MASK_THRESHOLD
    log.info(f'    Brain mask : {brain_mask.sum():,} / {n_vox:,} voxels')

    # Apply sitk.MaskImageFilter per subject
    mask_3d = brain_mask.reshape(ref_shape).astype(np.uint8)
    for i, row in tqdm(df.iterrows(), total=len(df), desc='    sitk.MaskImageFilter'):
        arr    = gmv_matrix[i].reshape(ref_shape)
        masked = apply_brain_mask_sitk(arr, mask_3d)
        gmv_matrix[i] = masked.flatten()

    # Voxel-wise z-score across subjects
    log.info('    Voxel-wise z-scoring...')
    gmv_masked_z = voxelwise_zscore(gmv_matrix[:, brain_mask])
    log.info(f'    GMV matrix (z-scored, masked): {gmv_masked_z.shape}')
    del gmv_matrix

    # ── [3] Load 20 gene maps via SimpleITK ──────────────────────────────
    log.info('\n[3] Loading 20 gene maps (SimpleITK resample)...')
    gene_vectors = {}   # gene_id -> 1D brain-masked array

    for gene_id, (symbol, group) in GENE_LIST.items():
        zip_path = os.path.join(GENE_MAPS_DIR, f'{gene_id}.zip')
        if not os.path.exists(zip_path):
            log.warning(f'    ZIP not found: {zip_path} ({symbol}) — skipping')
            gene_vectors[gene_id] = None
            continue
        vec = load_gene_map(zip_path, ref_sitk_image, ref_shape, brain_mask)
        if vec is None:
            log.warning(f'    Could not load gene {symbol} ({gene_id})')
            gene_vectors[gene_id] = None
        else:
            log.info(f'    Loaded {symbol:10s} (ID={gene_id:6d})  '
                     f'shape={vec.shape}  mean={vec.mean():.4f}')
            gene_vectors[gene_id] = vec

    # ── [4] Compute per-subject GAP scores (NMI matrix) ──────────────────
    log.info('\n[4] Computing per-subject GAP scores (NMI)...')
    log.info('    Shape will be (N_subjects x N_genes) = '
             f'({len(df)} x {len(GENE_LIST)})')

    # Build result rows: one row per subject
    records = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='    Subjects'):
        rec = {
            'subject_id': row.get('subject_id', i),
            'group':      row['group'],
        }
        subj_gmv = gmv_masked_z[i]   # 1D z-scored GMV for this subject

        for gene_id, (symbol, group) in GENE_LIST.items():
            gene_vec = gene_vectors.get(gene_id)
            if gene_vec is None or np.std(gene_vec) == 0:
                rec[symbol] = np.nan
            else:
                rec[symbol] = nmi_1d(subj_gmv, gene_vec)

        records.append(rec)

    gap_df = pd.DataFrame(records)

    # Save the full GAP matrix
    gap_csv = os.path.join(OUTPUT_DIR, 'gap_matrix_20genes.csv')
    gap_df.to_csv(gap_csv, index=False)
    log.info(f'    Saved GAP matrix: {gap_csv}')

    # Quick preview
    gene_symbols = [v[0] for v in GENE_LIST.values()]
    log.info('\n    Mean NMI per gene (AD / CN):')
    ad_mask = gap_df['group'] == 'AD'
    cn_mask = gap_df['group'] == 'CN'
    for sym in gene_symbols:
        ad_mean = gap_df.loc[ad_mask, sym].mean()
        cn_mean = gap_df.loc[cn_mask, sym].mean()
        log.info(f'      {sym:10s}  AD={ad_mean:.4f}  CN={cn_mean:.4f}')

    # ── [5] Independent t-test per gene (AD vs CN) ───────────────────────
    log.info('\n[5] Running independent t-tests (AD vs CN) per gene...')

    ttest_rows = []
    for gene_id, (symbol, gene_group) in GENE_LIST.items():
        ad_vals = gap_df.loc[ad_mask, symbol].dropna().values
        cn_vals = gap_df.loc[cn_mask, symbol].dropna().values

        if len(ad_vals) < 3 or len(cn_vals) < 3:
            log.warning(f'    {symbol}: not enough data for t-test')
            continue

        t_stat, p_val = stats.ttest_ind(ad_vals, cn_vals, equal_var=False)
        d             = cohens_d(ad_vals, cn_vals)

        ttest_rows.append({
            'gene_id':     gene_id,
            'gene_symbol': symbol,
            'gene_group':  gene_group,
            't_stat':      t_stat,
            'p_value':     p_val,
            'cohens_d':    d,
            'ad_mean_nmi': ad_vals.mean(),
            'cn_mean_nmi': cn_vals.mean(),
            'ad_n':        len(ad_vals),
            'cn_n':        len(cn_vals),
        })

    ttest_df = pd.DataFrame(ttest_rows)

    # FDR correction across all 20 genes
    reject, p_fdr, _, _ = multipletests(
        ttest_df['p_value'].values, alpha=0.05, method='fdr_bh'
    )
    ttest_df['p_fdr']          = p_fdr
    ttest_df['fdr_significant'] = reject

    # Sort: AD-related first, then by p-value
    ttest_df = ttest_df.sort_values(
        ['gene_group', 'p_value'],
        ascending=[True, True]         # 'AD-related' < 'Not related' alphabetically
    ).reset_index(drop=True)

    ttest_csv = os.path.join(OUTPUT_DIR, 'ttest_results_20genes.csv')
    ttest_df.to_csv(ttest_csv, index=False)
    log.info(f'    Saved t-test results: {ttest_csv}')

    # Print results table
    log.info('\n    T-test results:')
    log.info(f'    {"Gene":<10} {"Group":<12} {"t":>7} {"p":>8} {"q(FDR)":>8} {"d":>6} {"Sig":>5}')
    log.info('    ' + '-' * 62)
    for _, row in ttest_df.iterrows():
        sig = 'YES' if row['fdr_significant'] else '-'
        log.info(
            f'    {row["gene_symbol"]:<10} {row["gene_group"]:<12} '
            f'{row["t_stat"]:>7.3f} {row["p_value"]:>8.4f} '
            f'{row["p_fdr"]:>8.4f} {row["cohens_d"]:>6.3f} {sig:>5}'
        )

    # -- [6] Plots
    log.info("\n[6] Generating plots...")
    log.info("    Saving individual boxplots (one per gene, named by gene ID)...")
    plot_all_boxplots(gap_df, ttest_df, OUTPUT_DIR)
    log.info("    Saving volcano plot...")
    plot_volcano(ttest_df, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    n_sig = ttest_df['fdr_significant'].sum()
    n_sig_ad  = ttest_df[ttest_df['gene_group'] == 'AD-related']['fdr_significant'].sum()
    n_sig_neg = ttest_df[ttest_df['gene_group'] == 'Not related']['fdr_significant'].sum()

    log.info('\n' + '=' * 60)
    log.info('SUMMARY')
    log.info('=' * 60)
    log.info(f'  Total genes tested    : {len(ttest_df)}')
    log.info(f'  FDR significant       : {n_sig}')
    log.info(f'    AD-related genes    : {n_sig_ad} / 10 significant')
    log.info(f'    Negative controls   : {n_sig_neg} / 10 significant')
    log.info(f'  Outputs in            : {OUTPUT_DIR}')
    log.info('=' * 60)


if __name__ == '__main__':
    main()