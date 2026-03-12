#!/usr/bin/env python
# -*- coding: utf-8 -*-

# =============================================================================
# CellNeighborEX v2 end-to-end pipeline
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc

# --- CellNeighborEX modules ---
from CellNeighborEX.ccisignal import (
    qc_filter_mt_genes,
    preprocess_reference,
    train_regression_model,
    validate_model_training,
    export_reference_signatures,
    prepare_spatial_data,
    train_deconvolution_model,
    export_deconvolution_results,
    compute_proportion,
    cluster_spots_by_proportion,
    plot_deconvolution_result,
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


# ---------------------------
# Argument parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="CellNeighborEX v2 end-to-end: ccisignal → ccigenes → ccipairs"
    )

    # Positional inputs (optional when resuming from step 6)
    parser.add_argument(
        "scRNA_seq_file", type=str, nargs="?",
        help="Path to scRNA-seq h5ad file (annotated reference)"
    )
    parser.add_argument(
        "Visium_file", type=str, nargs="?",
        help="Path to Visium h5ad file"
    )

    # I/O
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")

    # Resume / precomputed paths
    parser.add_argument(
        "--start_from", type=int, choices=[1, 6], default=1,
        help="Start pipeline from step 1 (default) or step 6 (use precomputed outputs)."
    )
    parser.add_argument(
        "--ref_signatures_file", type=str, default=None,
        help="Path to precomputed reference signatures h5ad "
             "(default: {outdir}/reference_signatures/sc_ccisignal.h5ad)"
    )
    parser.add_argument(
        "--deconv_file", type=str, default=None,
        help="Path to precomputed deconvolution h5ad "
             "(default: {outdir}/cell2location_map/sp_ccisignal.h5ad)"
    )

    # ccisignal options
    parser.add_argument("--label_key", type=str, default="predictions",
                        help="Column in adata_ref.obs with cell type labels")
    parser.add_argument("--ref_batch_key", type=str, default=None,
                        help="Optional batch column in adata_ref.obs")
    parser.add_argument("--ref_do_filtering", action="store_true",
                        help="Apply gene filtering before RegressionModel training")
    parser.add_argument("--ref_max_epochs", type=int, default=250,
                        help="Max epochs for RegressionModel (reference signature learning)")
    parser.add_argument("--sp_batch_key", type=str, default=None,
                        help="Optional batch column in adata_vis.obs")
    parser.add_argument("--sp_max_epochs", type=int, default=4000,
                        help="Max epochs for Cell2location deconvolution")
    parser.add_argument("--n_cells_per_location", type=int, default=10,
                        help="Expected average number of cells per Visium spot")
    parser.add_argument("--detection_alpha", type=float, default=20.0,
                        help="Hyperparameter controlling RNA detection normalization")
    parser.add_argument("--sp_lr", type=float, default=0.002,
                        help="Learning rate for Cell2location")
    parser.add_argument("--skip_qc_mt", action="store_true",
                        help="Skip mitochondrial gene filtering on Visium")
    parser.add_argument("--do_cluster", action="store_true",
                        help="Optionally perform Leiden clustering on proportions")

    # ccigenes options
    parser.add_argument("--cluster_info", type=str, default="proportion_leiden",
                        help="obs column containing cluster labels (default: proportion_leiden)")
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

    # ccipairs options
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge (L2) regularization strength for NB preselection step")
    parser.add_argument("--self_interaction", action="store_true",
                        help="Allow self × self interactions")
    parser.add_argument("--pval_term", type=float, default=0.05,
                        help="p-value cutoff for individual NB terms")

    return parser.parse_args()


# ---------------------------
# Helper: small auto batch-size (if you later want to add GPU logic)
# ---------------------------
def _auto_batch_size(n_obs: int, use_gpu: bool):
    """Return batch_size heuristic (kept simple; currently unused in training calls)."""
    if not use_gpu:
        return None  # full-batch on CPU
    bs = max(256, n_obs // 20)
    return int(min(1024, bs))


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sc.settings.figdir = args.outdir

    # Precomputed paths handling
    # - If starting from steps 1–5: allow defaults under --outdir
    # - If starting from step 6   : require explicit file paths
    if args.start_from == 6:
        if args.ref_signatures_file is None:
            raise ValueError("You must provide --ref_signatures_file when using --start_from=6")
        if args.deconv_file is None:
            raise ValueError("You must provide --deconv_file when using --start_from=6")
        ref_sig_path = args.ref_signatures_file
        deconv_path = args.deconv_file
    else:
        default_ref_sig = os.path.join(args.outdir, "reference_signatures", "sc_ccisignal.h5ad")
        default_deconv  = os.path.join(args.outdir, "cell2location_map", "sp_ccisignal.h5ad")
        ref_sig_path = args.ref_signatures_file or default_ref_sig
        deconv_path  = args.deconv_file or default_deconv

    # These variables will hold the final paths to print at the end
    out_ref_path = None
    out_deconv_path = None

    # --------------------------------
    # Step 1–5: full run (or) resume
    # --------------------------------
    if args.start_from == 1:
        # Step 1: Load raw inputs
        if args.scRNA_seq_file is None or args.Visium_file is None:
            raise ValueError("When --start_from=1, positional arguments scRNA_seq_file and Visium_file are required.")

        adata_ref = sc.read(args.scRNA_seq_file)
        adata_vis = sc.read(args.Visium_file)

        # Step 2: Mitochondrial QC on Visium (optional)
        if not args.skip_qc_mt:
            if "SYMBOL" not in adata_vis.var.columns:
                adata_vis.var["SYMBOL"] = adata_vis.var_names
            adata_vis = qc_filter_mt_genes(adata_vis, gene_symbol_col="SYMBOL", remove_mt=True)

        # Step 3: Train reference regression (cell-type signatures)
        adata_ref = preprocess_reference(
            adata_ref,
            label_key=args.label_key,
            batch_key=args.ref_batch_key,
            do_filtering=args.ref_do_filtering,
        )
        ref_model = train_regression_model(adata_ref, max_epochs=args.ref_max_epochs, gpu=True)
        validate_model_training(ref_model, fig_path=os.path.join(args.outdir, "ref_regression_elbo_history.pdf"))
        ref_dir = os.path.join(args.outdir, "reference_signatures")
        adata_ref = export_reference_signatures(ref_model, adata_ref, result_dir=ref_dir, filename="sc_ccisignal.h5ad")
        out_ref_path = os.path.join(ref_dir, "sc_ccisignal.h5ad")
        adata_ref = sc.read(out_ref_path)
        
        # Step 4: Prepare Visium + run deconvolution (Cell2location)
        adata_vis, inf_aver = prepare_spatial_data(
            adata_vis, adata_ref, factor_key="mod", ensure_unique=True, add_sample=True
        )
        sp_model = train_deconvolution_model(
            adata_vis, inf_aver,
            batch_key=args.sp_batch_key,
            n_cells_per_location=args.n_cells_per_location,
            detection_alpha=args.detection_alpha,
            max_epochs=args.sp_max_epochs,
            lr=args.sp_lr,
            gpu=True,
            verbose=True,
        )
        sp_dir = os.path.join(args.outdir, "cell2location_map")
        adata_vis = export_deconvolution_results(
            sp_model, adata_vis, result_dir=sp_dir, prefix="sp2", filename="sp_ccisignal.h5ad"
        )
        out_deconv_path = os.path.join(sp_dir, "sp_ccisignal.h5ad")
        adata_vis = sc.read(out_deconv_path)
        
        # Save ELBO history for the spatial model
        validate_model_training(sp_model, fig_path=os.path.join(args.outdir, "query_deconvolution_elbo_history.pdf"))

        # Step 5: Compute proportions and, optionally, cluster
        adata_vis = compute_proportion(adata_vis)
        if args.do_cluster:
            adata_vis = cluster_spots_by_proportion(adata_vis, n_neighbors=7, resolution=0.2)

        # Optionally generate quick-look plots (safe to keep)
        plot_deconvolution_result(adata_vis, output_path="showovc_abundance.png", mode="abundance", cell_limit=11)
        plot_deconvolution_result(adata_vis, output_path="showovc_proportion.png", mode="proportion", cell_limit=11)

    else:
        # Resume mode: directly load precomputed outputs for Step 6+
        if not os.path.exists(ref_sig_path):
            raise FileNotFoundError(
                f"Precomputed reference signatures not found: {ref_sig_path}. "
                f"Use --ref_signatures_file to provide a valid path."
            )
        if not os.path.exists(deconv_path):
            raise FileNotFoundError(
                f"Precomputed deconvolution h5ad not found: {deconv_path}. "
                f"Use --deconv_file to provide a valid path."
            )

        print("[Resume] Loading precomputed files:")
        print(f"- Reference signatures: {ref_sig_path}")
        print(f"- Deconvolution map   : {deconv_path}")

        adata_ref = sc.read(ref_sig_path)
        adata_vis = sc.read(deconv_path)

        # Recompute proportion columns (idempotent; will overwrite if already present)
        adata_vis = compute_proportion(adata_vis)

        # Optional clustering from precomputed deconvolution
        if args.do_cluster:
            adata_vis = cluster_spots_by_proportion(adata_vis, n_neighbors=7, resolution=0.2)

        # For consistent final printout
        out_ref_path = ref_sig_path
        out_deconv_path = deconv_path

    # --------------------------------
    # Step 6: ccigenes — tests & merge
    # --------------------------------

    # Optional gene symbol simplification for both objects
    if args.species:
        simplify_gene_names(adata_ref, args.species)
        simplify_gene_names(adata_vis, args.species)

    # Retrieve factor (cell-type) names from cell2location's 'mod' slot in adata_ref.uns
    factor_names = adata_ref.uns["mod"]["factor_names"]

    # Get the average cell-type signatures (robust to varm/var storage)
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
                "Missing expected signature columns in adata_ref.var: "
                + ", ".join(missing)
            )
        inf_aver2 = adata_ref.var[expected_cols].copy()
        inf_aver2.columns = factor_names

    # Intersect genes and subset both matrices consistently
    intersect = np.intersect1d(adata_vis.var_names, inf_aver2.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver2 = inf_aver2.loc[intersect, :].copy()

    # Compute expected expression using cell2location posterior means stored in adata_vis.uns['mod']
    total_df = (
        adata_vis.obs[adata_vis.uns["mod"]["factor_names"]]
        @ inf_aver2.T
        * adata_vis.uns["mod"]["post_sample_means"]["m_g"]
    )
    final_df = (
        (total_df + adata_vis.uns["mod"]["post_sample_means"]["s_g_gene_add"][0])
        * adata_vis.uns["mod"]["post_sample_means"]["detection_y_s"]
    )

    # Build observed / expected DataFrames with metadata
    meta_data = adata_vis.obs.copy()
    observed_expression = pd.concat([adata_vis.to_df(), meta_data], axis=1)
    expected_expression = pd.concat([final_df, meta_data], axis=1)

    observed_expression["condition"] = "observed"
    expected_expression["condition"] = "expected"

    # Make unique indices to avoid clashes when concatenating
    observed_expression.index = [f"{idx}_before" for idx in observed_expression.index]
    expected_expression.index = [f"{idx}_after" for idx in expected_expression.index]

    combined_df = pd.concat([observed_expression, expected_expression])
    drop_cols = list(meta_data.columns) + ["condition"]

    # AnnData with only gene matrix in X; group/cluster in .obs
    adata = sc.AnnData(X=combined_df.drop(columns=drop_cols))
    adata.obs["condition"] = combined_df["condition"].values

    # Ensure the cluster column exists (from clustering or provided column)
    cluster_info = args.cluster_info
    if cluster_info not in combined_df.columns:
        raise ValueError(
            f"Cluster column '{cluster_info}' not found. "
            f"Enable --do_cluster (creates 'proportion_leiden') or provide a valid --cluster_info present in Visium .obs."
        )
    adata.obs[cluster_info] = combined_df[cluster_info].astype("category")

    # Clean column names for downstream modeling
    observed_expression = clean_column_names(observed_expression)
    expected_expression = clean_column_names(expected_expression)
    meta_data = clean_column_names(meta_data)
    inf_aver2 = clean_column_names(inf_aver2)

    # Get cleaned cell-type names from adata_vis.uns['mod']['factor_names']
    cell_types = obtain_clean_celltype_names(adata_vis)

    # Permutation test across clusters
    permutation_results = permutation_test_all_clusters(
        adata,
        cluster_info=cluster_info,
        observed_expression=observed_expression,
        expected_expression=expected_expression,
        n_permutations=args.n_permutations,
        use_zeros=args.perm_use_zeros,
        random_seed=42,
        path=args.outdir,
    )

    # Chi-square goodness-of-fit test
    de_results_chi = chi_square_goodness_of_fit(
        adata,
        cluster_info=cluster_info,
        groupby="condition",
        reference="observed",
        target="expected",
        use_zeros=args.chi_use_zeros,
    )

    # Merge results and compute combined p-values (ACAT)
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

    # Final filtering for niche genes
    final_results = merged[
        (merged["combined_p_value_adj"] < 0.01) &
        (merged["mean_ref"] > merged["mean_tgt"]) &
        (merged["logfc"] > args.log_fc)
    ].copy()
    final_results.to_csv(os.path.join(args.outdir, "ccigenes_results.csv"), index=False)

    # --------------------------------
    # Step 7: ccipairs — interaction modeling & comparison
    # --------------------------------

    # Build cluster summary from normalized proportions
    norm_deconv = meta_data.loc[:, cell_types].div(meta_data.loc[:, cell_types].sum(axis=1), axis=0)
    norm_deconv[cluster_info] = meta_data[cluster_info]
    cluster_summary = norm_deconv.groupby(cluster_info).mean()
    cluster_summary.loc["mean"] = cluster_summary.mean()

    # Ridge-regularized NB selection then individual NB per interaction term
    all_models, gene_analysis = regress_residual_on_interaction_with_regularization(
        observed_expression,
        expected_expression,
        celltypes=cell_types,
        cell_sig=inf_aver2,
        niche_gene_results=final_results,
        cluster_summary=cluster_summary,
        cluster_info=cluster_info,
        self_interaction=args.self_interaction,
        use_zeros=False,
        alpha=args.alpha,
    )

    # Extract significant interaction terms and keep top-5 per (cluster, gene)
    all_interaction_terms = extract_interaction_terms(
        all_models, gene_analysis, p_value_threshold=args.pval_term
    )
    all_interaction_terms = all_interaction_terms.groupby(["cluster", "gene"]).apply(
        lambda x: x.sort_values(by="p_value").head(5)
    ).reset_index(drop=True)

    # Optional: compare with curated interaction databases
    if args.species:
        all_interaction_terms = compare_database(query_df=all_interaction_terms, species=args.species)
    else:
        print("Species not specified. Skipping database comparison.")

    all_interaction_terms.to_csv(os.path.join(args.outdir, "ccipairs_results.csv"), index=False)

    # --------------------------------
    # Final prints
    # --------------------------------
    print("Pipeline finished successfully.")
    if out_ref_path is not None:
        print(f"- Reference signatures: {out_ref_path}")
    if out_deconv_path is not None:
        print(f"- Deconvolution map   : {out_deconv_path}")
    # print(f"- All genes (tests)   : {os.path.join(args.outdir, 'allgenes_results.csv')}")
    print(f"- CCI genes         : {os.path.join(args.outdir, 'ccigenes_results.csv')}")
    print(f"- CCI pairs         : {os.path.join(args.outdir, 'ccipairs_results.csv')}")


if __name__ == "__main__":
    main()