import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cell2location
from cell2location.models import RegressionModel, Cell2location
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
import warnings
warnings.filterwarnings('ignore')


# QC: Filter mitochondrial genes
def qc_filter_mt_genes(adata, gene_symbol_col="SYMBOL", mt_prefix="MT-", remove_mt=True):
    """
    Identify and optionally remove mitochondrial genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing raw or preprocessed counts.

    gene_symbol_col : str
        Column in adata.var that stores gene symbols.

    mt_prefix : str
        Prefix used to identify mitochondrial genes.
    
    remove_mt : bool
        If True, mitochondrial genes are removed from the dataset.
        If False, only flagging is performed without removal.    

    Returns
    -------
    AnnData
        Updated AnnData with 'MT_gene' flag in var and 
        mitochondrial counts stored in obsm['MT'].
    """
    
    # Calculate QC metrics including total counts and % mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Flag genes that are mitochondrial by prefix
    mt_prefixes = ["MT-", "MT_"]
    adata.var['MT_gene'] = [
        any(gene.upper().startswith(prefix) for prefix in mt_prefixes)
        for gene in adata.var[gene_symbol_col]
    ]

    # Save mitochondrial gene counts in obsm for reference
    adata.obsm['MT'] = adata[:, adata.var['MT_gene'].values].X.toarray()
    
    # Remove mitochondrial genes from the dataset
    if remove_mt:
        adata = adata[:, ~adata.var['MT_gene'].values]
    
    return adata


# Prepare scRNA-seq reference
def preprocess_reference(adata_ref, label_key='predictions', 
                         batch_key=None,
                         do_filtering=False,
                         cell_count_cutoff=5, 
                         cell_percentage_cutoff2=0.03, 
                         nonz_mean_cutoff=1.12):
    """
    Prepare scRNA-seq reference AnnData for cell2location regression model.

    Parameters
    ----------
    adata_ref : AnnData
        Annotated scRNA-seq reference dataset.

    label_key : str
        Column name in adata_ref.obs that contains cell type annotations.
   
    batch_key : str or None
        Column in adata_ref.obs to correct for batch effects.
        If None, no batch correction is applied.    

    do_filtering : bool
        If True, apply gene filtering before training.

    cell_count_cutoff : int
        Minimum number of cells expressing a gene to retain. (default 5)

    cell_percentage_cutoff2 : float
        Minimum fraction of cells expressing a gene to retain. (default 0.03)

    nonz_mean_cutoff : float
        Minimum non-zero mean expression across cells for a gene to be kept. (default 1.12)

    Returns
    -------
    AnnData
        Preprocessed AnnData object ready for cell2location.
    """
    
    # Store gene names in SYMBOL column
    adata_ref.var['SYMBOL'] = adata_ref.var.index

    # Optionally filter genes to reduce noise
    if do_filtering:
        from cell2location.utils.filtering import filter_genes
        selected = filter_genes(
            adata_ref,
            cell_count_cutoff=cell_count_cutoff,
            cell_percentage_cutoff2=cell_percentage_cutoff2,
            nonz_mean_cutoff=nonz_mean_cutoff
        )
        # Keep only selected genes
        adata_ref = adata_ref[:, selected].copy()
    
    # Setup AnnData for regression model training
    RegressionModel.setup_anndata(adata=adata_ref, batch_key=batch_key, labels_key=label_key)
    
    return adata_ref


# Train regression model
def train_regression_model(adata_ref, max_epochs=250, batch_size=2500, lr=0.002, train_size=1, gpu=False, verbose=True):
    """
    Train a regression model using scRNA-seq reference data.

    Parameters
    ----------
    adata_ref : AnnData
        Annotated scRNA-seq reference dataset used to learn cell-type-specific expression signatures.

    max_epochs : int
        Number of training epochs.

    batch_size : int
        Mini-batch size. None disables mini-batching.

    lr : float
        Learning rate for the optimizer.

    train_size : float
        Fraction of dataset used for training.

    gpu : bool
        Whether to use GPU acceleration.

    verbose : bool
        Whether to print training history status.

    Returns
    -------
    RegressionModel
        Trained regression model object.
    """
    
    # Initialize regression model with reference data
    model = RegressionModel(adata_ref)
    
    # Train regression model with provided hyperparameters
    model.train(max_epochs=max_epochs, use_gpu=gpu, lr=lr, train_size=train_size, batch_size=batch_size)
    
    # Print training history if verbose is True
    if verbose:
        print(model.history_ if model.history_ else "Warning: ELBO loss history is empty.")
    
    return model


# Validate model training
def validate_model_training(model, fig_path=None):
    """
    Plot ELBO loss curve to visually validate regression or deconvolution model training.

    Parameters
    ----------
    model : RegressionModel or Cell2location
        Trained model object.

    fig_path : str or None
        If provided, save the ELBO curve as PDF/PNG.
    """
    
    # Warn if history is empty (training may have failed)
    if not model.history_:
        print("Warning: ELBO loss history is empty.")
    else:
        # Plot training history
        with plt.rc_context():
            model.plot_history(0)
            plt.legend(labels=['full data training'])
            # Save figure if path provided
            if fig_path:
                plt.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close()


# Export regression results
def export_reference_signatures(model, adata_ref, result_dir, filename="sc_cell2loc.h5ad", gpu=False):
    """
    Export posterior cell-type expression estimates from the trained regression model.

    Parameters
    ----------
    model : RegressionModel
        Trained regression model.

    adata_ref : AnnData
        scRNA-seq reference dataset.

    result_dir : str
        Directory where model and h5ad file will be saved.
        
    filename : str
        Output file name for the h5ad file. 
    
    gpu : bool
        Whether to use GPU for posterior sampling.       

    Returns
    -------
    AnnData
        Updated AnnData with exported posterior.
    """
    
    # Export posterior distribution of expression profiles
    adata_ref = model.export_posterior(
        adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500, 'use_gpu': gpu}
    )
    
    # Ensure results directory exists
    os.makedirs(result_dir, exist_ok=True)
    
    # Save trained regression model
    model.save(result_dir, overwrite=True)
    
    # Clear raw slot to reduce file size
    adata_ref.raw = None
    
    # Save updated AnnData object
    adata_ref.write(os.path.join(result_dir, filename))
    
    return adata_ref


# Prepare Visium data
def prepare_spatial_data(adata_vis, adata_ref, factor_key="mod", ensure_unique=True, add_sample=True):
    """
    Filter and align Visium spatial data and cell-type expression signatures.

    Parameters
    ----------
    adata_vis : AnnData
        Visium spatial transcriptomics dataset.

    adata_ref : AnnData
        Trained reference AnnData containing inferred cell-type signatures.

    factor_key : str
        Key to access factor names in adata_ref.uns (usually "mod").

    ensure_unique : bool
        If True, ensure var_names are unique.

    add_sample : bool
        If True, create 'sample' column in obs using keys from adata_vis.uns['spatial'].

    Returns
    -------
    Tuple[AnnData, DataFrame]
        Filtered Visium data and aligned expression signature matrix.
    """

    # Add 'sample' column if missing, using keys from spatial metadata
    if add_sample and "sample" not in adata_vis.obs.columns:
        adata_vis.obs['sample'] = list(adata_vis.uns['spatial'].keys())[0]

    # Add SYMBOL column for genes if not already present
    if "SYMBOL" not in adata_vis.var.columns:
        adata_vis.var['SYMBOL'] = adata_vis.var_names

    # Ensure gene names are unique to avoid conflicts
    if ensure_unique:
        adata_vis.var_names_make_unique()

    # Extract inferred average expression per cell type
    inf_aver = (
        adata_ref.varm["means_per_cluster_mu_fg"]
        if "means_per_cluster_mu_fg" in adata_ref.varm
        else adata_ref.var[[f"means_per_cluster_mu_fg_{i}" for i in adata_ref.uns[factor_key]["factor_names"]]]
    )
    inf_aver.columns = adata_ref.uns[factor_key]["factor_names"]

    # Find intersection of genes between Visium and reference
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    
    # Subset both Visium and reference signatures to common genes
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    return adata_vis, inf_aver


# Train deconvolution model
def train_deconvolution_model(adata_vis, inf_aver, batch_key=None, batch_size=None, 
                              n_cells_per_location=10, detection_alpha=20,
                              max_epochs=4000, lr=0.002, gpu=False, verbose=True):
    """
    Train cell2location spatial deconvolution model.

    Parameters
    ----------
    adata_vis : AnnData
        Preprocessed Visium dataset.

    inf_aver : pd.DataFrame
        Signature matrix of cell-type average expression.

    batch_key : str
        Column in adata_vis.obs denoting sample identity.

    n_cells_per_location : int
        Expected cells per Visium spot.

    detection_alpha : float
        Hyperparameter for RNA detection normalization.

    max_epochs : int
        Training epochs.
        
   lr : float
        Learning rate for the optimizer (default 0.002).     

    gpu : bool
        Whether to use GPU.

    verbose : bool
        Print ELBO loss after training.

    Returns
    -------
    Cell2location
        Trained deconvolution model.
    """
    
    # Setup AnnData object for deconvolution
    Cell2location.setup_anndata(adata=adata_vis, batch_key=batch_key)
    
    # Initialize deconvolution model
    model = Cell2location(
        adata_vis, cell_state_df=inf_aver,
        N_cells_per_location=n_cells_per_location,
        detection_alpha=detection_alpha,
    )
    
    # Train model with specified hyperparameters
    model.train(max_epochs=max_epochs, batch_size=batch_size, train_size=1, use_gpu=gpu, max_steps=max_epochs, lr=lr)
    
    # Print training history
    if verbose:
        print(model.history_ if model.history_ else "Warning: ELBO loss history is empty.")
    
    return model


# Export deconvolution results
def export_deconvolution_results(model, adata_vis, result_dir, prefix="sp2", filename=None, gpu=False):
    """
    Export posterior results from trained spatial deconvolution model.
    This version additionally computes expected expression per cell type
    and saves only one file (default: sp2_cell2loc.h5ad).

    Parameters
    ----------
    model : Cell2location
        Trained deconvolution model.

    adata_vis : AnnData
        Visium data.

    result_dir : str
        Directory for saving results.

    prefix : str
        Prefix for the saved h5ad file (default: "sp2").

    filename : str or None
        Custom output file name. If None, defaults to f"{prefix}_cell2loc.h5ad".
        
    gpu : bool
        Whether to use GPU for posterior sampling.    

    Returns
    -------
    AnnData
        Updated Visium AnnData with posterior results (including expected expression).
    """
    
    # Export posterior cell abundance estimates
    adata_vis = model.export_posterior(
        adata_vis, sample_kwargs={"num_samples": 1000, "batch_size": adata_vis.n_obs, "use_gpu": gpu}
    )
    
    # Compute expected expression per cell type and store in .layers
    expected_dict = model.module.model.compute_expected_per_cell_type(
        model.samples["post_sample_q05"], model.adata_manager
    )
    for i, n in enumerate(model.factor_names_):
        sanitized_name = n.replace("/", "_")
        adata_vis.layers[sanitized_name] = expected_dict["mu"][i]
    
    # Create results directory if missing
    os.makedirs(result_dir, exist_ok=True)
    
    # Save trained model
    model.save(result_dir, overwrite=True)
    
    # Use default filename if none provided
    if filename is None:
        filename = f"{prefix}_cell2loc.h5ad"   # -> sp2_cell2loc.h5ad by default
    
    # Save updated Visium AnnData object
    adata_vis.write(os.path.join(result_dir, filename))
    
    return adata_vis


# Compute cell proportions
def compute_proportion(adata_vis):
    """
    Normalize cell abundance to cell-type proportions.

    Parameters
    ----------
    adata_vis : AnnData
        Visium AnnData with abundance estimates.

    Returns
    -------
    AnnData
        Updated with proportions in obsm['proportion'].
    """
    
    # Retrieve factor names (cell types)
    cell_types = adata_vis.uns["mod"]["factor_names"]
    
    # Load abundance values into obs
    adata_vis.obs[cell_types] = adata_vis.obsm["q05_cell_abundance_w_sf"]
    
    # Normalize abundances to sum to 1 per spot
    adata_vis.obs[cell_types] = (adata_vis.obs[cell_types].T / adata_vis.obs[cell_types].sum(axis=1)).T
    
    # Store proportions in obsm for later use
    adata_vis.obsm["proportion"] = adata_vis.obs[cell_types]
    
    # Reset obs to absolute abundance values
    adata_vis.obs[cell_types] = adata_vis.obsm["q05_cell_abundance_w_sf"]
    
    return adata_vis


# Compute expected expression
def compute_expected_expression(model, adata_vis):
    """
    Compute expected expression per cell type and store in layers.

    Parameters
    ----------
    model : Cell2location
        Trained deconvolution model.

    adata_vis : AnnData
        Visium data.

    Returns
    -------
    AnnData
        With expected expression in layers.
    """
    
    # Compute expected gene expression for each cell type
    expected_dict = model.module.model.compute_expected_per_cell_type(
        model.samples["post_sample_q05"], model.adata_manager
    )
    
    # Store each cell type’s expression in layers, replacing "/" with "_"
    for i, n in enumerate(model.factor_names_):
        sanitized_name = n.replace("/", "_")
        adata_vis.layers[sanitized_name] = expected_dict["mu"][i]
    
    return adata_vis


# Plot results
def plot_deconvolution_result(adata_vis, output_path, mode='abundance', cell_limit=11,
                              figsize=(4.5, 5), cmap='magma', vmin=0, vmax='p99.2'):
    """
    Plot spatial maps of abundance or proportion.

    Parameters
    ----------
    adata_vis : AnnData
        Visium data with results.

    output_path : str
        Filename to save the figure.

    mode : {'abundance', 'proportion'}
        Type of plot to generate.

    cell_limit : int
        Number of cell types to display.

    figsize : tuple
        Figure size for the plot (default: (4.5, 5)).

    cmap : str
        Colormap used for visualization (default: 'magma').

    vmin : float or str
        Minimum value for colormap scaling (default: 0).

    vmax : float or str
        Maximum value for colormap scaling 
        (can use 'pXX.X' to set percentile, e.g., 'p99.2').
    """

    # Choose whether to plot proportions or abundances
    if mode == 'proportion':
        adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['proportion']
    else:
        adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

    # Select the first available sample
    sample_name = list(set(adata_vis.obs['sample']))[0]
    slide = adata_vis[adata_vis.obs['sample'] == sample_name, :].copy()

    # Create spatial plot with user-defined visualization parameters
    with mpl.rc_context({'axes.facecolor': 'black', 'figure.figsize': figsize}):
        sc.pl.spatial(
            slide,
            cmap=cmap,
            color=adata_vis.uns['mod']['factor_names'][:cell_limit],
            ncols=4,
            size=1.3,
            vmin=vmin,
            vmax=vmax,
            save=output_path
        )


# Cluster Visium spots
def cluster_spots_by_proportion(adata_vis, n_neighbors=7, resolution=0.2):
    """
    Perform clustering on Visium spots using proportions.

    Parameters
    ----------
    adata_vis : AnnData
        Visium AnnData.

    n_neighbors : int
        Number of neighbors in KNN graph.

    resolution : float
        Leiden clustering resolution.

    Returns
    -------
    AnnData
        With cluster assignments in obs['proportion_leiden'].
    """
    
    # Build KNN graph based on cell-type proportions
    sc.pp.neighbors(adata_vis, use_rep='proportion', n_neighbors=n_neighbors)
    
    # Run Leiden clustering
    sc.tl.leiden(adata_vis, resolution=resolution)
    
    # Save clustering labels in obs
    adata_vis.obs["proportion_leiden"] = adata_vis.obs["leiden"].astype("category")
    
    return adata_vis


# Select one slide (FOV)
def select_slide(adata, sample_name, batch_key="sample"):
    """
    Select a single slide (or field of view) from a multi-sample Visium object.

    Parameters
    ----------
    adata : AnnData
        Visium AnnData object with multiple slides.

    sample_name : str
        Name of the slide to select.

    batch_key : str
        Column in obs that identifies slides.

    Returns
    -------
    AnnData
        Subset AnnData containing only the specified slide.
    """
    
    # Subset obs to include only the chosen sample
    slide = adata[adata.obs[batch_key].isin([sample_name]), :].copy()
    
    # Retrieve the correct spatial key for the chosen sample
    s_keys = list(slide.uns["spatial"].keys())
    s_spatial = np.array(s_keys)[[sample_name in k for k in s_keys]][0]
    
    # Update uns['spatial'] to keep only the selected slide’s info
    slide.uns["spatial"] = {s_spatial: slide.uns["spatial"][s_spatial]}
    
    return slide