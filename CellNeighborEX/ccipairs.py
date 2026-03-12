from scipy.stats import spearmanr
from scipy.optimize import minimize
from statsmodels.genmod.families import NegativeBinomial
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .utils import load_database_files
import warnings
warnings.filterwarnings("ignore")


# Function for getting regression plots
def visualize_nb_regression(model, gene, cluster, save_path=None):
    """
    Visualizes the Negative Binomial Regression results for a given model.

    Parameters:
        model (statsmodels.genmod.generalized_linear_model.GLMResults): Trained Negative Binomial model.
        gene (str): Gene name associated with the model.
        cluster (str or int): Cluster ID associated with the gene.
        save_path (str, optional): Directory path to save the plots. If None, plots will be displayed without saving.
    """
   
    # Extract model coefficients
    coef_df = model.summary2().tables[1].reset_index().rename(columns={"index": "Feature"})

    # Select significant interaction terms (excluding the Intercept)
    significant_terms = coef_df[(coef_df["P>|z|"] < 0.05) & (coef_df["Feature"] != "Intercept")]
    if significant_terms.empty:
        #print(f"No significant interaction terms found for {gene} in cluster {cluster}. Skipping visualization.")
        return

    for _, row in significant_terms.iterrows():
        top_interaction = row["Feature"]  # Most significant interaction term
        #print(f"Using {top_interaction} as the interaction term for visualization of {gene} in cluster {cluster}.")

        # Retrieve X (interaction term abundance) and Y (Observed - Expected Gene Expression)
        x_values = model.model.exog[:, model.model.exog_names.index(top_interaction)]  # Abundance of the interaction term
        y_values = model.model.endog  # Observed - Expected gene expression

        # Scatter plot: Interaction term vs. residual gene expression
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=x_values, y=y_values, alpha=0.5, color="blue")  # Removed label to exclude legend

        # Generate the NB regression line
        x_sorted = np.linspace(min(x_values), max(x_values), 100)  # Define a wider range for X values

        # Retrieve all independent variables used in the model
        predictors = model.model.exog_names

        # Create a DataFrame for predictions (initialize with zeros for all variables)
        pred_df = pd.DataFrame({col: np.zeros(len(x_sorted)) for col in predictors})  
        pred_df[top_interaction] = x_sorted  # Assign actual values only to the selected interaction term

        # Perform prediction using the trained model
        y_pred = model.predict(pred_df)

        # Add the Negative Binomial regression line to the plot
        plt.plot(x_sorted, y_pred, color="red", linestyle="--")

        # Extract coefficient and p-value for the top interaction term
        coef_value = row["Coef."]
        coef_p = row["P>|z|"]

        # Construct regression equation (without "Regression: ")
        regression_eq = f"{coef_value:.2f} * {top_interaction} (p={coef_p:.2g})"

        # Add regression equation text to the plot (without border)
        plt.text(
            0.05, 0.95,  # Position (normalized coordinates)
            regression_eq,
            fontsize=10, color="black", ha="left", va="top", transform=plt.gca().transAxes
        )

        # Customize the plot
        plt.xlabel(f"{top_interaction} abundance", fontsize=14)
        plt.ylabel(f"Observed - Expected {gene} (Cluster {cluster})", fontsize=14)
        plt.grid(False)
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
            scatter_plot_path = os.path.join(save_path, f"{gene}_cluster{cluster}_{top_interaction}_nb_regression.pdf")
            plt.savefig(scatter_plot_path, format="pdf", bbox_inches="tight")
            print(f"Saved: {scatter_plot_path}")
        else:
            plt.show()

        plt.close()


# Function for annotating niche genes with sources and/or targets in databases
def compare_database(query_df: pd.DataFrame, species: str, verbose: bool = False) -> pd.DataFrame:
    """
    Compare gene-level findings in `query_df` with curated interaction databases
    (OmniPath: source-target; CellChat/CellTalk: ligand-receptor) and annotate
    per-row flags for (i) which DB(s) contain the gene and (ii) whether a
    partner gene appears in the opposite direction within the same cluster.

    Parameters
    ----------
    query_df : pd.DataFrame
        Must contain columns:
          - 'gene' (str)
          - 'cluster' (str/int)
          - 'index_celltype' (str)
          - 'neighboring_celltype' (str)
    species : str
        One of {'human', 'mouse', 'rat'} (case-insensitive).
    verbose : bool
        If True, print lightweight warnings for missing files/columns.

    Returns
    -------
    pd.DataFrame
        Input `query_df` with added columns:
          - 'source_omnipath', 'pair_omnipath'
          - 'source_cellchat', 'pair_cellchat'
          - 'source_celltalk', 'pair_celltalk'
        (only for DBs available for the species)
    """

    # --- 0) Basic validation and normalization
    if not isinstance(query_df, pd.DataFrame) or query_df.empty:
        if verbose:
            print("[compare_database] Empty or invalid query_df. Nothing to annotate.")
        return query_df

    species_key = (species or "").strip().lower()
    db_index = load_database_files()  # {'human': {'omnipath': path, ...}, ...}

    if species_key not in db_index:
        if verbose:
            print(f"[compare_database] Unsupported species: '{species}'. Skipping DB comparison.")
        return query_df

    species_files = db_index[species_key]  # e.g., {'omnipath': '/...csv', 'cellchat': '/...csv', ...}

    # --- 1) Prepare annotation columns (only for DBs that exist for this species)
    db_names = [k for k in ['omnipath', 'cellchat', 'celltalk'] if k in species_files]
    for db in db_names:
        query_df[f"source_{db}"] = "unknown"
        query_df[f"pair_{db}"] = "unknown"

    # If no DB is available for the species, nothing to do
    if not db_names:
        if verbose:
            print(f"[compare_database] No databases registered for species '{species_key}'.")
        return query_df

    # --- 2) Normalize gene symbols for matching: `_` -> `-`, uppercase
    # Keep original gene column intact; use a temporary normalized column for matching
    query_df = query_df.copy()
    query_df["gene_normalized"] = (
        query_df["gene"].astype(str).str.replace("_", "-", regex=False).str.upper()
    )

    # --- 3) Load databases safely; normalize columns and keep only (source, target)
    # Expected columns in CSV: 'source_gene_symbol', 'target_gene_symbol'
    # If the file is missing or columns are absent, we simply skip that DB.
    all_databases = {}
    for db_name in db_names:
        file_path = species_files.get(db_name, None)
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            if verbose:
                print(f"[compare_database] Skip '{db_name}': file not found -> {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            if verbose:
                print(f"[compare_database] Skip '{db_name}': cannot read CSV -> {file_path} ({e})")
            continue

        # Check required columns
        required_cols = {"source_gene_symbol", "target_gene_symbol"}
        if not required_cols.issubset(set(df.columns)):
            if verbose:
                print(f"[compare_database] Skip '{db_name}': required columns missing in {file_path}")
            continue

        # Normalize symbols in DB the same way as queries
        db_df = df.loc[:, ["source_gene_symbol", "target_gene_symbol"]].copy()
        for col in ["source_gene_symbol", "target_gene_symbol"]:
            db_df[col] = (
                db_df[col].astype(str).str.replace("_", "-", regex=False).str.upper()
            )

        all_databases[db_name] = db_df

    if not all_databases:
        if verbose:
            print("[compare_database] No usable database loaded; returning input.")
        query_df.drop(columns=["gene_normalized"], inplace=True)
        return query_df

    # --- 4) Row-wise lookup: mark source DB and pair/single status per DB
    # For each row (gene, cluster, index_celltype, neighboring_celltype),
    #   - if DB contains gene as source or target, mark source_DB = db_name
    #   - then check if the "partner gene" appears in the same cluster with
    #     swapped index/neighbor cell types -> mark as 'pair'; else 'single'.
    #
    # Note: We keep the first 'pair' terms first, then 'single' terms in a '; ' list.
    for idx, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Processing Queries"):
        gene_norm = row["gene_normalized"]
        cluster = row["cluster"]
        index_ct = row["index_celltype"]
        neighbor_ct = row["neighboring_celltype"]

        for db_name, db_df in all_databases.items():
            # Candidate rows in DB where gene is either source or target
            matched = db_df[
                (db_df["source_gene_symbol"] == gene_norm) |
                (db_df["target_gene_symbol"] == gene_norm)
            ]

            if matched.empty:
                continue

            # Mark presence in this DB
            query_df.at[idx, f"source_{db_name}"] = db_name

            pair_list, single_list = [], []

            # For each match, derive the partner gene and check "pair" condition
            for _, m in matched.iterrows():
                src = m["source_gene_symbol"]
                tgt = m["target_gene_symbol"]

                if gene_norm == src:
                    partner = tgt
                    label = f"{gene_norm}-{partner}"
                else:
                    partner = src
                    label = f"{partner}-{gene_norm}"

                # Pair condition: same cluster AND opposite index/neighbor cell types
                pair_cond = query_df[
                    (query_df["cluster"] == cluster) &
                    (query_df["gene"].astype(str).str.replace("_", "-", regex=False).str.upper() == partner) &
                    (query_df["index_celltype"] == neighbor_ct) &
                    (query_df["neighboring_celltype"] == index_ct)
                ]

                if not pair_cond.empty:
                    pair_list.append(f"pair {label}")
                else:
                    single_list.append(f"single {label}")

            # Store pairs first, then singles
            query_df.at[idx, f"pair_{db_name}"] = "; ".join(pair_list + single_list) if (pair_list or single_list) else "unknown"

    # --- 5) Cleanup temp column and return
    query_df.drop(columns=["gene_normalized"], inplace=True)
    return query_df


# Function for extracting significant interaction terms    
def extract_interaction_terms(models_per_gene, gene_analysis, p_value_threshold=0.05):
    """
    Extracts significant interaction terms from regression models and retrieves their 
    associated cell types based on stored analysis results.

    Parameters:
        models_per_gene (dict): Dictionary where keys are genes, values are dictionaries of clusters
                                containing fitted regression models.
        gene_analysis (dict): Dictionary storing analysis results, including index_celltype and 
                              neighboring_celltype for each interaction term.
        p_value_threshold (float, optional): Maximum allowable p-value for inclusion (default: 0.05).

    Returns:
        pd.DataFrame: Extracted interaction terms with the following columns:
                      - 'gene': Gene name.
                      - 'cluster': Cluster ID.
                      - 'term': Interaction term.
                      - 'coef': Coefficient value.
                      - 'std_err': Standard error.
                      - 'p_value': P-value.
                      - 'index_celltype': Primary cell type.
                      - 'neighboring_celltype': Secondary cell type.
    """
    
    results = []

    for gene, clusters in models_per_gene.items():
        for cluster, models in clusters.items():
            # Check if individual models exist
            if "individual_models" not in models or not models["individual_models"]:
                print(f"Skipping {gene} in cluster {cluster} due to missing models.")
                continue  

            for interaction, model in models["individual_models"].items():
                if not hasattr(model, "summary2"):  # Ensure it's a statsmodels model
                    print(f"Skipping {interaction} in {gene}, {cluster} (Invalid model type)")
                    continue
                
                # Extract p-values
                summary = model.summary2().tables[1]
            
                # Retrieve precomputed index_celltype and neighboring_celltype
                index_neighbor_info = gene_analysis.get(gene, {}).get(cluster, {}).get('index_neighbor_info', {})

                for term, row in summary.iterrows():
                    if term != 'Intercept' and row['Coef.'] > 0 and term in index_neighbor_info:
                        p_value = row['P>|z|']
                        coef = row['Coef.']
                        std_err = row['Std.Err.']
                        

                        # Retrieve precomputed cell type pair
                        index_celltype, neighbor_celltype = index_neighbor_info[term]

                        results.append([
                            gene, cluster, term, index_celltype, neighbor_celltype, 
                            coef, std_err, p_value
                        ])

    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=[
        'gene', 'cluster', 'term', 'index_celltype', 'neighboring_celltype',
        'coef', 'std_err', 'p_value'
    ])
    
    # Apply filtering criteria
    if p_value_threshold is not None:
        results_df = results_df[results_df['p_value'] <= p_value_threshold]
        
    # Ensure 'p_value' column is float type
    results_df['p_value'] = results_df['p_value'].astype(float)

    # Sort results by 'cluster', 'gene', and ascending 'p_value'
    results_df = results_df.sort_values(by=['cluster', 'gene', 'p_value'], ascending=[True, True, True])

    return results_df


# Function for customizing Ridge (L2) regularized non-negative negative binomial regression
def ridge_neg_binomial_nonnegative(formula, data, alpha=1.0):
    """
    Custom Ridge Regularized Non-Negative Negative Binomial Regression.

    Parameters:
        formula (str): Model formula for regression.
        data (pd.DataFrame): Input dataset containing independent and dependent variables.
        alpha (float): Regularization strength for Ridge regression (L2 penalty).

    Returns:
        np.array: Optimized non-negative regression coefficients.
    """
    # Fit an initial Negative Binomial regression model
    model = smf.glm(formula=formula, data=data, family=NegativeBinomial()).fit()

    def ridge_loss(params, model, alpha):
        """
        Computes the loss function for Ridge-regularized Negative Binomial regression.

        Parameters:
            params (array-like): Model parameters (regression coefficients).
            model (statsmodels GLMResults): Fitted GLM model.
            alpha (float): Regularization strength.

        Returns:
            float: Negative log-likelihood with L2 penalty.
        """
        loglike = model.llf  # Compute log-likelihood
        penalty = alpha * np.sum(params**2)  # L2 regularization term (Ridge)
        return -loglike + penalty  # Maximizing log-likelihood, hence negative sign

    # Define bounds to enforce non-negative coefficients
    bounds = [(0, None) for _ in range(len(model.params))]  # Set to be greater than or equal to 0

    # Optimize the loss function with non-negative constraint
    opt_result = minimize(ridge_loss, model.params, args=(model, alpha), method='L-BFGS-B', bounds=bounds)
    
    return opt_result.x  # Return optimized non-negative regression coefficients


# Main regression function using Ridge regularized non-negative NB regression
def regress_residual_on_interaction_with_regularization(observed_expression, expected_expression, 
                                                        celltypes, cell_sig, niche_gene_results,
                                                        cluster_summary, cluster_info='proportion_leiden', 
                                                        self_interaction=False, use_zeros=False, 
                                                        alpha=1.0):
    """
    Performs Ridge-regularized Negative Binomial Regression on interaction terms.
    Then, for each selected term, a separate Negative Binomial Regression is run without Ridge regularization.

    Returns:
        models_per_gene (dict): Dictionary of regression models per gene and cluster.
        gene_analysis (dict): Dictionary storing coefficients, p-values, VIFs, and cell type information.
    """
    
    # Define genes by filtering only those present in cell_sig
    genes = [gene for gene in niche_gene_results['gene'].unique() if gene in cell_sig.index]
    
    # Initialize dictionaries to store models and analysis results
    models_per_gene = {}
    gene_analysis = {}

    for gene in tqdm(genes, desc="Processing Genes", unit="gene"):
        valid_gene = gene.replace("-", "_").replace(".", "_")  # Ensure valid variable names
        clusters = list(niche_gene_results.loc[niche_gene_results['gene'] == valid_gene, 'cluster'])
        models_per_cluster = {}
        analysis_per_cluster = {}

        for cluster in clusters:
            sub_observed = observed_expression.loc[observed_expression[cluster_info] == cluster, :]
            sub_expected = expected_expression.loc[expected_expression[cluster_info] == cluster, :]
            sub_deconv = sub_observed.loc[:, celltypes]  # cell abundances

            # Compute residuals
            residuals = sub_observed[valid_gene].values - sub_expected[valid_gene].values
            
            # Compute scaled residuals for each gene
            # The standard residual is calculated as the difference between observed and expected expression.
            # To calibrate for the tendency that expected expression values increase with observed values, 
            # we divide by the square root of the expected expression. This helps to adjust the effect of large expected values.
            residuals_scaled = sub_observed[valid_gene].values - (sub_expected[valid_gene].values / np.sqrt(sub_expected[valid_gene].values))

            if not use_zeros:
                positive_indices = residuals > 0
                sub_observed = sub_observed.loc[positive_indices, :]
                sub_deconv = sub_deconv.loc[positive_indices, :]
                residuals = residuals[positive_indices]
                residuals_scaled = residuals_scaled[positive_indices]
                
            # Compute cell type-specific expression contributions
            cell_exp = sub_deconv.multiply(cell_sig.loc[valid_gene, :], axis=1)
            total_cell_exp = cell_exp.sum(axis=1)
            
            # Store computed values for each cell type
            cell_type_values = {cell_type: [] for cell_type in celltypes}
            # Compute cell type-specific contributions per spot
            for spot_idx in range(sub_observed.shape[0]):
                obs_value = sub_observed.iloc[spot_idx][valid_gene]
                if total_cell_exp.iloc[spot_idx] > 0:
                    for cell_type in celltypes:
                        cell_type_value = (
                            (obs_value / total_cell_exp.iloc[spot_idx]) * 
                            sub_deconv.iloc[spot_idx, sub_deconv.columns.get_loc(cell_type)] * 
                            cell_sig.loc[valid_gene, cell_type]
                        )
                        cell_type_values[cell_type].append(cell_type_value)
            
            # Store correlation results for each cell type
            correlation_results = []
            
            for cell_type, values in cell_type_values.items():
                cell_type_array = np.array(values)
            
                if len(cell_type_array) == len(residuals_scaled):
                    corr, pvalue = spearmanr(cell_type_array, residuals_scaled)
                    correlation_results.append((cell_type, corr, pvalue))
                else:
                    print(f"Length mismatch for {cell_type}: {len(cell_type_array)} vs {len(residuals)}")
                    correlation_results.append((cell_type, np.nan, np.nan))
            
            # Sort correlation results by correlation value (descending)
            sorted_results = sorted(
                [res for res in correlation_results if not np.isnan(res[1]) and res[2] < 0.05], 
                key=lambda x: x[1], 
                reverse=True
            )

            # Select cell types with correlation >= 0.6 as index_celltype candidates
            candidates = [res[0] for res in sorted_results if res[1] >= 0.6]
            
            if len(candidates) > 0:
                # Generate interaction terms
                interaction_terms = []
                unique_pairs = set()
                index, niche = np.meshgrid(candidates, celltypes)
                interaction_pairs = np.array([index.flatten(), niche.flatten()]).T
                
                index_neighbor_info = {}
            
                for pair in interaction_pairs:
                    unique_pairs.add(tuple(sorted(pair)))
                interaction_pairs = np.array(list(unique_pairs))
    
                for (f1, f2) in interaction_pairs:
                    if self_interaction or f1 != f2:
                        interaction_term = f'{f1}_{f2}'
                        interaction_terms.append(interaction_term)
                        sub_deconv[interaction_term] = sub_deconv[f1] * sub_deconv[f2] 
                        
                        # Determine index_celltype and neighboring_celltype
                        if (f1 in candidates) and (f2 not in candidates):
                            index_celltype = f1
                            neighbor_celltype = f2
                        elif (f2 in candidates) and (f1 not in candidates):
                            index_celltype = f2
                            neighbor_celltype = f1
                        elif (f1 in candidates) and (f2 in candidates):
                            # Get correlation coefficient values for f1 and f2 from sorted_results
                            corr_f1 = next((res[1] for res in sorted_results if res[0] == f1), -1)
                            corr_f2 = next((res[1] for res in sorted_results if res[0] == f2), -1)
                        
                            # Assign based on the higher correlation coefficient
                            if corr_f1 >= corr_f2:
                                index_celltype = f1
                                neighbor_celltype = f2
                            else:
                                index_celltype = f2
                                neighbor_celltype = f1
        
                        index_neighbor_info[interaction_term] = (index_celltype, neighbor_celltype)
    
                sub_deconv[valid_gene] = residuals  # Set residuals as dependent variable
                
                # Apply Ridge Regularization to Negative Binomial Regression
                if len(interaction_terms) >= 1:
                    formula = f"{valid_gene} ~ {' + '.join(interaction_terms)}"
                    try:
                        ridge_coeffs = ridge_neg_binomial_nonnegative(formula, sub_deconv, alpha=alpha)
                        intercept = ridge_coeffs[0]
                        ridge_coeffs_adj = ridge_coeffs[1:]
                        
                        # Compute new residuals for each variable separately
                        individual_model_results = {}  # Store models per variable
                        individual_coefficients = {}  # Store coefficients per variable                   
                        
                        for variable in interaction_terms:
                            try:
                                # Compute adjusted residual: remove all other independent variables
                                other_terms = [v for v in interaction_terms if v != variable]
                                sub_deconv[f"adjusted_residual_{variable}"] = (
                                    sub_deconv[valid_gene] - (intercept + np.dot(sub_deconv[other_terms], ridge_coeffs_adj[[interaction_terms.index(v) for v in other_terms]]))
                                )
    
                                # Perform Negative Binomial Regression for this single variable
                                single_formula = f"adjusted_residual_{variable} ~ {variable}"
                                single_model = smf.glm(formula=single_formula, data=sub_deconv, family=NegativeBinomial()).fit()
    
                                # Check if coefficient is positive 
                                if single_model.params[variable] > 0:
                                    # Store results
                                    individual_model_results[variable] = single_model
                                    individual_coefficients[variable] = single_model.params[variable]                               
    
                            except Exception as e:
                                print(f"Skipping single regression for {valid_gene}, variable {variable}, cluster {cluster}: {e}")
                                continue    
    
                        # Store models and results
                        models_per_cluster[cluster] = {
                            "ridge_model": ridge_coeffs,
                            "individual_models": individual_model_results
                        }
                        analysis_per_cluster[cluster] = {
                            "ridge_coefficients": ridge_coeffs_adj,
                            "new_coefficients": individual_coefficients,
                            "index_neighbor_info": index_neighbor_info
                        }
    
                    except Exception as e:
                        print(f"Skipping ridge regression for {valid_gene} in cluster {cluster}: {e}")
                        continue    

        models_per_gene[gene] = models_per_cluster
        gene_analysis[gene] = analysis_per_cluster  

    return models_per_gene, gene_analysis

