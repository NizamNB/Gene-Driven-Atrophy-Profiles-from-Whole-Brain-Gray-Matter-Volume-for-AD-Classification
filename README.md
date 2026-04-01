# Gene-Driven-Atrophy-Profiles-from-Whole-Brain-Gray-Matter-Volume-for-AD-Classification
This project presents a novel, biologically-informed framework for classifying AD from structural MRI by integrating neuroimaging with spatial gene expression data. Instead of treating brain atrophy as purely structural, the approach incorporates molecular context to better capture the mechanisms driving regional vulnerability.

## Motivation

Structural MRI, particularly T1-weighted imaging, enables voxel-based morphometry (VBM) to quantify gray matter volume (GMV) across the whole brain. AD is associated with characteristic atrophy patterns in regions such as:

Medial temporal lobe
Hippocampus
Entorhinal cortex
Posterior cingulate
Association cortices

While these patterns are useful for classification, traditional models ignore the genetic and molecular processes that influence why specific regions degenerate.

Genes such as APOE, TREM2, CLU, BIN1, and APP are known to affect amyloid processing, neuroinflammation, lipid metabolism, and synaptic function. Importantly, gene expression varies spatially across the brain—offering an opportunity to directly link molecular drivers with observed atrophy patterns.

## Key Idea

If a gene contributes to neurodegeneration in AD, then its spatial expression pattern should correlate with the spatial distribution of gray matter loss.

This project leverages that principle by aligning:

Individual whole-brain GMV maps (from MRI)
Spatial gene expression maps

to identify biologically meaningful imaging-genetics relationships.

## Methodology
**Stage 1:** Gene–Atrophy Association
Extract GMV (Gray Matter Volume) using the spm and cat12 tool.
Compute whole-brain gap scores between each subject’s GMV map and each gene’s expression map.
Identify genes that show statistically significant spatial correspondence with AD-related atrophy patterns.
This step filters out irrelevant genes and highlights biologically meaningful candidates.
**Stage 2:** Gene-Driven Atrophy Profiles (GDAPs)
Construct Gene-Driven Atrophy Profiles (GDAPs) using only the significant genes.
Each GDAP represents how strongly a subject’s atrophy pattern aligns with each gene’s spatial expression.
These GDAPs are treated as multi-channel volumetric inputs.
**Stage 3:** Deep Learning Classification
Train a 3D Convolutional Neural Network (CNN) using GDAPs as input.
Perform binary classification:
Alzheimer’s Disease (AD)
Cognitively Normal (CN)
## Contributions
Introduces a multi-modal fusion framework combining neuroimaging and transcriptomics.
Moves beyond purely structural MRI analysis by incorporating biological interpretability.
Proposes GDAPs as a novel representation of gene-informed brain atrophy.
Demonstrates how spatial gene expression can guide deep learning models for improved disease classification.
## Tech Stack
* Matlab
* SPM12 tool
* CAT12
* Python
* PyTorch / TensorFlow
* Neuroimaging tools (SPM, SimpleITK, or Nilearn)
* Gene expression datasets (e.g., MedUni Vienna Predicted mRNA Maps)

## Acknowledgments
* Public neuroimaging datasets (e.g., OASIS-2)
* MedUni Vienna Predicted mRNA Maps
* Open-source neuroimaging and machine learning communities
