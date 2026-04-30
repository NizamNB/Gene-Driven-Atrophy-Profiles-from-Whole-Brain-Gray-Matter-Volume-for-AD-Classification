# =========================
# FINAL REGION-BASED NMI PIPELINE WITH PLOTS
# =========================

import os, zipfile, tempfile, warnings, logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import mutual_info_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = r'C:\Users\StujenskeLab\Documents\NAN151_workspace\Alzheimer'
CSV_PATH = os.path.join(BASE_DIR, 'subjects_labels.csv')
GENE_MAPS_DIR = r'U:\Research\AZ\meduni_gene_maps'
ATLAS_PATH = os.path.join(BASE_DIR, 'AAL.nii')
OUTPUT_DIR = os.path.join(BASE_DIR, 'final_NMI_results')

BRAIN_MASK_THRESHOLD = 0.1
MIN_VOXELS = 30

# =========================
# GENE LIST
# =========================
GENE_LIST = {
    348:'APOE',54209:'TREM2',10347:'ABCA7',6653:'SORL1',23607:'CD2AP',
    1191:'CLU',274:'BIN1',114815:'SORCS1',123041:'SLC24A4',2185:'PTK2B',
    672:'BRCA1',324:'APC',1080:'CFTR',3043:'HBB',3625:'INHBB',
    4625:'MYH7',4080:'PAX6',5172:'SLC26A4',5251:'PHEX',3861:'KRT14',
    344901:'OSTN',29992:'PILRA',9325:'TRIP4',91584:'PLXNA4',
    55063:'ZCWPW1',3635:'INPP5D',4208:'MEF2C',7162:'TPBG',
    10142:'AKAP9',3127:'HLA-DRB5',8633:'UNC5C',8301:'PICALM',
    51225:'ABI3',122618:'PLD4',51338:'MS4A4A',5664:'PSEN2',
    94241:'TP53INP1',5663:'PSEN1',5157:'PDGFRL',9619:'ABCG1',
    965:'CD58',22986:'SORCS3',51809:'GALNT7',4137:'MAPT',
    23242:'COBL',57537:'SORCS2',10242:'KCNMB2',23646:'PLD3',
    945:'CD33',5336:'PLCG2',79746:'ECHDC3'
}

# =========================
# HELPERS
# =========================
def load_gmv(path):
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, arr.transpose(2,1,0)

def resample_to_ref(path, ref):
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(sitk.sitkLinear)
    return sitk.GetArrayFromImage(r.Execute(img)).astype(np.float32).transpose(2,1,0)

def load_gene(zip_path, ref_img):
    with zipfile.ZipFile(zip_path, 'r') as z:
        nii = [n for n in z.namelist() if n.endswith('.nii')][0]
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(nii, tmp)
            return resample_to_ref(os.path.join(tmp, nii), ref_img)

def load_atlas(ref_img, gmv_3d):
    atlas_img = sitk.ReadImage(ATLAS_PATH, sitk.sitkFloat32)
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor)

    atlas = sitk.GetArrayFromImage(r.Execute(atlas_img)).astype(np.int32)
    atlas = atlas.transpose(2,1,0)

    atlas[gmv_3d < BRAIN_MASK_THRESHOLD] = 0

    ids = np.unique(atlas)
    ids = ids[ids > 0]

    return atlas, ids

def region_means(vol, atlas, ids):
    return np.array([
        np.nanmean(vol[atlas == r]) if (atlas == r).sum() > MIN_VOXELS else np.nan
        for r in ids
    ])

# =========================
# TRUE NMI
# =========================
def compute_nmi(x, y, bins=20):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 10:
        return np.nan

    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    x_bin = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_bin = np.digitize(y, np.histogram_bin_edges(y, bins=bins))

    mi = mutual_info_score(x_bin, y_bin)
    hx = mutual_info_score(x_bin, x_bin)
    hy = mutual_info_score(y_bin, y_bin)

    if hx == 0 or hy == 0:
        return np.nan

    return mi / np.sqrt(hx * hy)

# =========================
# PLOTTING
# =========================
def plot_single_boxplot(symbol, ad_vals, cn_vals, p_value, p_fdr, d, output_dir):
    plt.figure(figsize=(4,5))

    data = pd.DataFrame({
        'Group': ['AD']*len(ad_vals) + ['CN']*len(cn_vals),
        'Value': np.concatenate([ad_vals, cn_vals])
    })

    sns.boxplot(x='Group', y='Value', data=data)
    sns.stripplot(x='Group', y='Value', data=data, color='black', alpha=0.4)

    plt.title(f"{symbol}\np={p_value:.2e}, FDR={p_fdr:.2e}, d={d:.2f}")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{symbol}.png"))
    plt.close()

def plot_grid(gap_df, ttest_df, output_dir):
    genes = ttest_df['gene']
    cols = 5
    rows = int(np.ceil(len(genes)/cols))

    plt.figure(figsize=(cols*4, rows*4))

    for i, gene in enumerate(genes):
        plt.subplot(rows, cols, i+1)

        ad = gap_df[gap_df['group']=='AD'][gene]
        cn = gap_df[gap_df['group']=='CN'][gene]

        data = pd.DataFrame({
            'Group': ['AD']*len(ad) + ['CN']*len(cn),
            'Value': np.concatenate([ad, cn])
        })

        sns.boxplot(x='Group', y='Value', data=data)
        plt.title(gene)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_all_genes_grid.png"))
    plt.close()

def plot_volcano(ttest_df, output_dir):
    plt.figure(figsize=(6,6))

    x = ttest_df['effect']
    y = -np.log10(ttest_df['p_value'])

    plt.scatter(x, y, alpha=0.7)

    for _, row in ttest_df.iterrows():
        if row['p_fdr'] < 0.05:
            plt.text(row['effect'], -np.log10(row['p_value']), row['gene'])

    plt.axhline(-np.log10(0.05), linestyle='--')

    plt.xlabel("Effect Size (AD - CN)")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano Plot")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volcano_plot.png"))
    plt.close()

# =========================
# MAIN
# =========================
def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    ref_img, ref_arr = load_gmv(df['gm_file'].iloc[0])
    atlas, region_ids = load_atlas(ref_img, ref_arr)

    # --- DIAGNOSTIC BLOCK ---
    print(f"\nTotal Shen-268 regions with >{MIN_VOXELS} voxels after brain masking: {len(region_ids)}")
    print(f"\nRegion labels that survived:\n{region_ids.tolist()}")
    pd.DataFrame({'region_label': region_ids}).to_csv(
        os.path.join(OUTPUT_DIR, 'surviving_region_labels.csv'), index=False
    )
    # ------------------------

    log.info(f"Shen-268 regions loaded: {len(region_ids)}")

    # Load gene maps
    gene_maps = {}
    for gid, sym in GENE_LIST.items():
        path = os.path.join(GENE_MAPS_DIR, f"{gid}.zip")
        if os.path.exists(path):
            gene_maps[sym] = load_gene(path, ref_img)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        _, gmv = load_gmv(row['gm_file'])

        rec = {"subject": row['subject_id'], "group": row['group']}

        for sym, gene in gene_maps.items():
            gmv_r = region_means(gmv, atlas, region_ids)
            gene_r = region_means(gene, atlas, region_ids)
            rec[sym] = compute_nmi(gmv_r, gene_r)

        records.append(rec)

    gap_df = pd.DataFrame(records)
    gap_df.to_csv(os.path.join(OUTPUT_DIR, "gap_matrix.csv"), index=False)

    # =========================
    # STATS
    # =========================
    results = []

    for sym in gene_maps.keys():
        ad = gap_df[gap_df['group']=="AD"][sym].dropna()
        cn = gap_df[gap_df['group']=="CN"][sym].dropna()

        if len(ad) < 3 or len(cn) < 3:
            continue

        t, p = stats.ttest_ind(ad, cn, equal_var=False)
        d = (ad.mean() - cn.mean()) / np.sqrt((ad.std()**2 + cn.std()**2)/2)

        results.append({
            "gene": sym,
            "AD_mean": ad.mean(),
            "CN_mean": cn.mean(),
            "p_value": p,
            "effect": ad.mean() - cn.mean(),
            "cohens_d": d
        })

    ttest_df = pd.DataFrame(results)
    ttest_df['p_fdr'] = multipletests(ttest_df['p_value'], method='fdr_bh')[1]

    ttest_df.to_csv(os.path.join(OUTPUT_DIR, "ttest_results.csv"), index=False)

    # =========================
    # PLOTS
    # =========================
    boxplot_dir = os.path.join(OUTPUT_DIR, "boxplots_per_gene")
    os.makedirs(boxplot_dir, exist_ok=True)

    for _, row in ttest_df.iterrows():
        sym = row['gene']

        plot_single_boxplot(
            sym,
            gap_df[gap_df['group']=="AD"][sym].dropna().values,
            gap_df[gap_df['group']=="CN"][sym].dropna().values,
            row['p_value'], row['p_fdr'], row['cohens_d'],
            boxplot_dir
        )

    plot_grid(gap_df, ttest_df, OUTPUT_DIR)
    plot_volcano(ttest_df, OUTPUT_DIR)

    print("\n=== DONE ===")
    print(f"Results saved in: {OUTPUT_DIR}")

# =========================
if __name__ == "__main__":
    main()