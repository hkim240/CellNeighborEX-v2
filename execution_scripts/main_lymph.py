# ==============================
# File: main_lymph.py
# Purpose: Run CellNeighborEX v2 Step 6+ by **combining two conditions**
#          (e.g., PBS and MS) that each already have precomputed
#          sc_ccisignal.h5ad (reference signatures) and
#          sp_ccisignal.h5ad (deconvolution map).
# ==============================

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc

# --- CellNeighborEX modules ---
from CellNeighborEX.ccisignal import (
    compute_proportion,
    cluster_spots_by_proportion,
)
from CellNeighborEX.ccigenes import (
    clean_column_names, obtain_clean_celltype_names,
    permutation_test_all_clusters, chi_square_goodness_of_fit,
    compute_combined_p_value, simplify_gene_names
)
from CellNeighborEX.ccipairs import (
    regress_residual_on_interaction_with_regularization,
    extract_interaction_terms, compare_database
)

# Python 3.9 compatibility for type hints
from typing import Optional, List


# ---------------------------
# Argument parser
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CellNeighborEX v2 (Step 6+) dual-condition combiner: "
            "Load precomputed sc_ccisignal/sp_ccisignal from PBS and MS, "
            "merge, and run ccigenes → ccipairs."
        )
    )

    # I/O
    parser.add_argument("--outdir", type=str, default="out_lymph", help="Output directory")

    # Required precomputed files (one pair per condition)
    parser.add_argument("--ref_signatures_file_PBS", type=str, required=True,
                        help="Path to PBS sc_ccisignal.h5ad (reference signatures)")
    parser.add_argument("--deconv_file_PBS", type=str, required=True,
                        help="Path to PBS sp_ccisignal.h5ad (deconvolution map)")

    parser.add_argument("--ref_signatures_file_MS", type=str, required=True,
                        help="Path to MS sc_ccisignal.h5ad (reference signatures)")
    parser.add_argument("--deconv_file_MS", type=str, required=True,
                        help="Path to MS sp_ccisignal.h5ad (deconvolution map)")

    # ccigenes options
    parser.add_argument("--cluster_info", type=str, default="spatial_kmeans",
                        help="obs column containing cluster labels (must exist in both Visium objects)")
    parser.add_argument("--log_fc", type=float, default=0.5,
                        help="Log2 fold-change threshold for niche gene filtering")
    parser.add_argument("--species", type=str, default=None,
                        help="Species for gene symbol simplification and DB comparison (human/mouse/rat)")

    parser.add_argument("--n_permutations", type=int, default=1000,
                        help="Number of permutations for permutation test")
    parser.add_argument("--perm_use_zeros", action="store_true",
                        help="Include zeros in permutation test")
    parser.add_argument("--chi_use_zeros", action="store_true",
                        help="Include zeros in chi-square test")

    # Optional: if the cluster column is missing and you want to generate it from proportions
    parser.add_argument("--do_cluster", action="store_true",
                        help="Perform Leiden clustering on proportions if cluster_info is absent")

    # ccipairs options
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge (L2) regularization strength for NB preselection step")
    parser.add_argument("--self_interaction", action="store_true",
                        help="Allow self × self interactions")
    parser.add_argument("--pval_term", type=float, default=0.05,
                        help="p-value cutoff for individual NB terms")

    return parser.parse_args()


# ---------------------------
# Utilities for Step 6 on a single condition
# ---------------------------

def _load_precomputed_pair(ref_sig_path: str, deconv_path: str,
                           species: Optional[str],
                           cluster_info: str,
                           do_cluster: bool,
                           condition_key_value: str):
    """
    Load sc_ccisignal (ref) and sp_ccisignal (Visium), compute proportions,
    optionally cluster, simplify gene names, and compute observed/expected
    matrices needed for downstream ccigenes/ccipairs.

    Returns
    -------
    adata_vis : AnnData (Visium)
    observed_expression : pd.DataFrame (genes + metadata)
    expected_expression : pd.DataFrame (genes + metadata)
    meta_data : pd.DataFrame (metadata only)
    inf_aver2 : pd.DataFrame (gene × factor signatures)
    cell_types_clean : list[str]
    """
    if not os.path.exists(ref_sig_path):
        raise FileNotFoundError(f"Reference signatures not found: {ref_sig_path}")
    if not os.path.exists(deconv_path):
        raise FileNotFoundError(f"Deconvolution map not found: {deconv_path}")

    adata_ref = sc.read(ref_sig_path)
    adata_vis = sc.read(deconv_path)

    # Tag sample key (standardize to 'sample_keys')
    adata_vis.obs["sample_keys"] = condition_key_value

    # Optional gene symbol simplification
    if species:
        simplify_gene_names(adata_ref, species)
        simplify_gene_names(adata_vis, species)

    # Compute/ensure proportions
    adata_vis = compute_proportion(adata_vis)

    # Cluster if requested or if cluster_info missing
    if (cluster_info not in adata_vis.obs.columns) and do_cluster:
        adata_vis = cluster_spots_by_proportion(adata_vis, n_neighbors=7, resolution=0.2)
    # After possible clustering, enforce category dtype if present
    if cluster_info in adata_vis.obs.columns:
        adata_vis.obs[cluster_info] = adata_vis.obs[cluster_info].astype("category")
    else:
        raise ValueError(
            f"Cluster column '{cluster_info}' not found in Visium obs for {condition_key_value}. "
            f"Provide an existing column or use --do_cluster."
        )

    # Factor names
    factor_names = adata_ref.uns["mod"]["factor_names"]

    # Pull the average signatures (robust to varm/var storage)
    if "means_per_cluster_mu_fg" in adata_ref.varm.keys():
        inf_aver2_raw = adata_ref.varm["means_per_cluster_mu_fg"]
        if isinstance(inf_aver2_raw, pd.DataFrame):
            inf_aver2 = inf_aver2_raw.copy()
        else:
            inf_aver2 = pd.DataFrame(
                inf_aver2_raw, index=adata_ref.var_names, columns=factor_names
            )
        inf_aver2.columns = factor_names
    else:
        expected_cols = [f"means_per_cluster_mu_fg_{i}" for i in factor_names]
        missing = [c for c in expected_cols if c not in adata_ref.var.columns]
        if missing:
            raise KeyError(
                "Missing expected signature columns in adata_ref.var: " + ", ".join(missing)
            )
        inf_aver2 = adata_ref.var[expected_cols].copy()
        inf_aver2.columns = factor_names

    # Intersect with Visium genes
    intersect = np.intersect1d(adata_vis.var_names, inf_aver2.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver2 = inf_aver2.loc[intersect, :].copy()

    # Expected expression using cell2location posterior means
    total_df = (
        adata_vis.obs[adata_vis.uns["mod"]["factor_names"]]
        @ inf_aver2.T
        * adata_vis.uns["mod"]["post_sample_means"]["m_g"]
    )
    final_df = (
        (total_df + adata_vis.uns["mod"]["post_sample_means"]["s_g_gene_add"][0])
        * adata_vis.uns["mod"]["post_sample_means"]["detection_y_s"]
    )

    # Observed/Expected with metadata
    meta_data = adata_vis.obs.copy()
    observed_expression = pd.concat([adata_vis.to_df(), meta_data], axis=1)
    expected_expression = pd.concat([final_df, meta_data], axis=1)

    observed_expression["condition"] = "observed"
    expected_expression["condition"] = "expected"

    # Make unique indices before concatenation across conditions later
    observed_expression.index = [f"{idx}_{condition_key_value}_before" for idx in observed_expression.index]
    expected_expression.index = [f"{idx}_{condition_key_value}_after" for idx in expected_expression.index]

    # Clean cell-type names from Visium model
    cell_types_clean = obtain_clean_celltype_names(adata_vis)

    return adata_vis, observed_expression, expected_expression, meta_data, inf_aver2, cell_types_clean


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sc.settings.figdir = args.outdir

    # Load & prep each condition
    (
        adata_PBS,
        observed_expression_PBS,
        expected_expression_PBS,
        meta_data_PBS,
        inf_aver_PBS,
        celltypes_PBS,
    ) = _load_precomputed_pair(
        args.ref_signatures_file_PBS,
        args.deconv_file_PBS,
        species=args.species,
        cluster_info=args.cluster_info,
        do_cluster=args.do_cluster,
        condition_key_value="PBS",
    )

    (
        adata_MS,
        observed_expression_MS,
        expected_expression_MS,
        meta_data_MS,
        inf_aver_MS,
        celltypes_MS,
    ) = _load_precomputed_pair(
        args.ref_signatures_file_MS,
        args.deconv_file_MS,
        species=args.species,
        cluster_info=args.cluster_info,
        do_cluster=args.do_cluster,
        condition_key_value="MS",
    )

    # Ensure cell types are aligned; if not, intersect
    if celltypes_PBS != celltypes_MS:
        common_ct = [ct for ct in celltypes_PBS if ct in set(celltypes_MS)]
        if len(common_ct) == 0:
            raise ValueError("No overlapping cell types between PBS and MS after cleaning.")
        celltypes = common_ct
        # Subset meta-data columns to common cell types for both
        for df_meta in (meta_data_PBS, meta_data_MS):
            missing = [c for c in celltypes if c not in df_meta.columns]
            if missing:
                raise ValueError(f"Missing expected cell-type columns in meta_data: {missing}")
    else:
        celltypes = celltypes_PBS

    # Align gene columns between conditions for observed/expected
    gene_cols_PBS = [c for c in observed_expression_PBS.columns if c not in meta_data_PBS.columns and c != "condition"]
    gene_cols_MS  = [c for c in observed_expression_MS.columns  if c not in meta_data_MS.columns  and c != "condition"]
    common_genes = sorted(list(set(gene_cols_PBS).intersection(set(gene_cols_MS))))
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between PBS and MS matrices.")

    # Reindex observed/expected to common genes (preserve metadata as-is)
    def _align_oe(df, meta_cols, common_genes):
        gene_part = df.loc[:, common_genes]
        rest_cols = [c for c in df.columns if c not in common_genes]
        return pd.concat([gene_part, df[rest_cols]], axis=1)

    observed_expression_PBS = _align_oe(observed_expression_PBS, meta_data_PBS.columns, common_genes)
    expected_expression_PBS = _align_oe(expected_expression_PBS, meta_data_PBS.columns, common_genes)
    observed_expression_MS  = _align_oe(observed_expression_MS,  meta_data_MS.columns,  common_genes)
    expected_expression_MS  = _align_oe(expected_expression_MS,  meta_data_MS.columns,  common_genes)

    # Align inf_aver as well
    common_genes_inf = sorted(list(set(inf_aver_PBS.index).intersection(set(inf_aver_MS.index)).intersection(set(common_genes))))
    if len(common_genes_inf) == 0:
        raise ValueError("No overlapping genes across inf_aver signatures and observed/expected.")
    inf_aver_PBS = inf_aver_PBS.loc[common_genes_inf, :]
    inf_aver_MS  = inf_aver_MS.loc[common_genes_inf, :]

    # --- Integration (user-specified recipe) ---
    # AnnData concat (Visium)
    adata = sc.concat([adata_PBS, adata_MS], join='inner', label='condition_group', keys=['PBS', 'MS'])
    adata.obs[args.cluster_info] = adata.obs[args.cluster_info].astype('category')

    # DataFrame concatenations
    observed_expression = pd.concat([observed_expression_PBS, observed_expression_MS])
    expected_expression = pd.concat([expected_expression_PBS, expected_expression_MS])
    meta_data = pd.concat([meta_data_PBS, meta_data_MS])

    # Build combined AnnData used by ccigenes tests
    combined_df = pd.concat([observed_expression, expected_expression])
    drop_cols = list(meta_data.columns) + ["condition"]

    adata_tests = sc.AnnData(X=combined_df.drop(columns=drop_cols))
    adata_tests.obs["condition"] = combined_df["condition"].values

    if args.cluster_info not in combined_df.columns:
        raise ValueError(
            f"Cluster column '{args.cluster_info}' not found after concatenation."
        )
    adata_tests.obs[args.cluster_info] = combined_df[args.cluster_info].astype("category")

    # Clean column names (downstream modeling safety)
    observed_expression = clean_column_names(observed_expression)
    expected_expression = clean_column_names(expected_expression)
    meta_data = clean_column_names(meta_data)

    # Choose a single cell-type signature matrix for ccipairs (they should match by design)
    # If both exist and differ slightly, take the average (optional). Here we take PBS version by default.
    inf_aver2 = clean_column_names(inf_aver_PBS.copy())

    # Cell types (already cleaned via obtain_clean_celltype_names for each; ensure present in meta_data)
    cell_types = [c for c in celltypes if c in meta_data.columns]
    if not cell_types:
        raise ValueError("No cell-type proportion columns found in meta_data after cleaning.")

    # --------------------------------
    # Step 6: ccigenes — tests & merge
    # --------------------------------
    permutation_results = permutation_test_all_clusters(
        adata_tests,
        cluster_info=args.cluster_info,
        observed_expression=observed_expression,
        expected_expression=expected_expression,
        n_permutations=args.n_permutations,
        use_zeros=args.perm_use_zeros,
        random_seed=42,
        path=args.outdir,
    )

    de_results_chi = chi_square_goodness_of_fit(
        adata_tests,
        cluster_info=args.cluster_info,
        groupby="condition",
        reference="observed",
        target="expected",
        use_zeros=args.chi_use_zeros,
    )

    merged = pd.merge(
        permutation_results[["gene", "cluster", "perm_p_value", "perm_p_value_adj"]],
        de_results_chi[[
            "gene", "cluster", "chi_stat", "chi_p_value", "chi_p_value_adj",
            "mean_ref", "mean_tgt", "logfc", "n_spots(observed > 0)", "n_spots(%)"
        ]],
        on=["gene", "cluster"],
    )
    merged["combined_p_value_adj"] = merged.apply(compute_combined_p_value, axis=1)
    # merged.to_csv(os.path.join(args.outdir, "allgenes_results.csv"), index=False)

    final_results = merged[
        (merged["combined_p_value_adj"] < 0.01) &
        (merged["mean_ref"] > merged["mean_tgt"]) &
        (merged["logfc"] > args.log_fc)
    ].copy()
    final_results.to_csv(os.path.join(args.outdir, "ccigenes_results.csv"), index=False)

    # --------------------------------
    # Step 7: ccipairs — interaction modeling & comparison
    # --------------------------------
    # Normalize proportions per spot and compute cluster-level means
    norm_deconv = meta_data.loc[:, cell_types].div(meta_data.loc[:, cell_types].sum(axis=1), axis=0)
    norm_deconv[args.cluster_info] = meta_data[args.cluster_info]
    cluster_summary = norm_deconv.groupby(args.cluster_info).mean()
    cluster_summary.loc["mean"] = cluster_summary.mean()

    all_models, gene_analysis = regress_residual_on_interaction_with_regularization(
        observed_expression,
        expected_expression,
        celltypes=cell_types,
        cell_sig=inf_aver2,
        niche_gene_results=final_results,
        cluster_summary=cluster_summary,
        cluster_info=args.cluster_info,
        self_interaction=args.self_interaction,
        use_zeros=False,
        alpha=args.alpha,
    )

    all_interaction_terms = extract_interaction_terms(
        all_models, gene_analysis, p_value_threshold=args.pval_term
    )
    all_interaction_terms = all_interaction_terms.groupby(["cluster", "gene"]).apply(
        lambda x: x.sort_values(by="p_value").head(5)
    ).reset_index(drop=True)

    if args.species:
        all_interaction_terms = compare_database(query_df=all_interaction_terms, species=args.species)

    all_interaction_terms.to_csv(os.path.join(args.outdir, "ccipairs_results.csv"), index=False)

    # --------------------------------
    # Final prints
    # --------------------------------
    print("Pipeline (dual-condition) finished successfully.")
    # print(f"- Merged All genes (tests) : {os.path.join(args.outdir, 'allgenes_results.csv')}")
    print(f"- CCI genes       : {os.path.join(args.outdir, 'ccigenes_results.csv')}")
    print(f"- CCI pairs       : {os.path.join(args.outdir, 'ccipairs_results.csv')}")


if __name__ == "__main__":
    main()