from scipy.stats import chisquare
import scipy as sp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# Function for applying Benjamini-Hochberg (FDR) correction
def adjust_p_values_bh(df, p_value_column, adjusted_column_name):
    """
    Apply Benjamini-Hochberg (FDR) correction to a DataFrame and add adjusted p-values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing p-values to be adjusted.

    p_value_column : str
        Name of the column in `df` containing the p-values to adjust.

    adjusted_column_name : str
        Name of the new column to store adjusted p-values.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with a new column for adjusted p-values.
    """
    
    # Convert column to numeric type (handle cases where p-values are stored as strings)
    df[p_value_column] = pd.to_numeric(df[p_value_column], errors='coerce')

    # Remove NaN values (multipletests does not handle NaNs properly)
    df = df.dropna(subset=[p_value_column]).copy()

    # Replace zero p-values with a very small number (to prevent log-related issues)
    df[p_value_column] = np.where(df[p_value_column] == 0, np.nextafter(0, 1), df[p_value_column])

    # Extract p-values as a NumPy array
    p_values = df[p_value_column].values

    # Perform Benjamini-Hochberg correction if p-values exist
    if len(p_values) > 0:
        _, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        df[adjusted_column_name] = adjusted_p_values
    else:
        df[adjusted_column_name] = np.nan  # Assign NaN if there are no p-values

    return df


# Function for performing chi-square goodness of fit test for each gene
def chi_square_goodness_of_fit(adata, cluster_info, groupby='condition', reference='observed', target='expected', use_zeros=True):
    """
    Perform Pearson's chi-square goodness of fit test for each gene in each cluster between two specified groups.
    
    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing the single-cell expression data.
        
    cluster_info : str
        The column name in `adata.obs` that contains cluster information.
        
    groupby : str
        The column name in `adata.obs` that contains the group information to compare (e.g., time points, conditions).
        
    reference : str
        The reference group label within the `groupby` column (e.g., observed expression).
        
    target : str
        The target group label within the `groupby` column to compare against the reference group (e.g., expected expression).
        
    use_zeros : bool, optional
        Whether to include zero values in the comparison (default is True).

    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the results of the chi-square goodness of fit test for each gene in each cluster.
    
    Example:
    --------
    results_df = chi_square_goodness_of_fit(adata, cluster_info='leiden', groupby='condition', 
                                  reference='control', target='treated', use_zeros=True)      
    """
    
    results = []
    clusters = adata.obs[cluster_info].cat.categories
    adata.var_names = adata.var_names.str.replace('-', '_') # Replace hyphens with underscores 
    adata.var_names = adata.var_names.str.replace('.', '_') # Replace dots with underscores 
    
    for cluster in clusters:
        print(f'Performing Peason\'s Chi-Square Test for cluster {cluster}...')
        sub_adata = adata[adata.obs[cluster_info] == cluster, ]
        
        # Copy the names of genes
        cluster_genes = adata.var_names.copy()
        
        for gene in tqdm(cluster_genes, desc='Chi-Square Test Progress'):
            gene_data = sub_adata[:, gene].X.flatten()
            ref_data = gene_data[sub_adata.obs[groupby] == reference]
            tgt_data = gene_data[sub_adata.obs[groupby] == target]

            # Handle zeros based on the use_zeros argument
            if not use_zeros:
                non_zero_indices = ref_data != 0
                ref_data = ref_data[non_zero_indices]
                tgt_data = tgt_data[non_zero_indices]

            tgt_data = tgt_data.astype(float)  # Ensure the data is float for calculations
            
            # Calculate the mean of the original tgt_data before normalization
            if len(ref_data) > 0 and len(tgt_data) > 0:
                mean_ref = np.mean(ref_data).astype(float)
                mean_tgt = np.mean(tgt_data).astype(float)  # Mean of original expected data (before normalization)
                
            # Remove or replace any zeros in the expected data to avoid division by zero
            tgt_data = tgt_data.copy()  # Make a copy of tgt_data to allow modifications
            tgt_data[tgt_data == 0] = 1e-10  # Replace zeros with a small value to avoid division by zero

            # Sum of observed and expected must be the same for chi-square test
            ref_sum = np.sum(ref_data)
            tgt_sum = np.sum(tgt_data)

            if tgt_sum > 0:  # Avoid division by zero
                tgt_data = tgt_data * (ref_sum / tgt_sum)  # Normalize tgt_data to match the sum of ref_data

            # Perform chi-square goodness of fit test
            min_count = 10
            if len(ref_data) >= min_count and len(tgt_data) >= min_count:
                stat, p_value = chisquare(f_obs=ref_data, f_exp=tgt_data)

                # Calculate log fold-change
                logfc = np.log2(mean_ref + 1) - np.log2(mean_tgt + 1)  # Avoid log(0)
                
                # Store results for all genes
                results.append([gene, float(stat), p_value, float(mean_ref), float(mean_tgt), float(logfc), cluster, len(ref_data), (len(ref_data)/len(sub_adata.obs_names))*100])

    # Create the DataFrame from the results list
    results_df = pd.DataFrame(results, columns=['gene', 'chi_stat', 'chi_p_value', 'mean_ref', 
                                                'mean_tgt', 'logfc', 'cluster', 'n_spots(observed > 0)', 'n_spots(%)'])
     
    # Benjamini-Hochberg correction
    results_df = adjust_p_values_bh(df=results_df, p_value_column='chi_p_value', adjusted_column_name='chi_p_value_adj')    
    
    return results_df


# Function for obtaining a combined p-value
def acat_test(pvalues, weights=None):
    '''acat_test()
    Aggregated Cauchy Assocaition Test
    A p-value combination method using the Cauchy distribution.
    
    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R
    
    Author: Ryan Neff
    
    Inputs:
        pvalues: <list or numpy array>
            The p-values you want to combine.
        weights: <list or numpy array>, default=None
            The weights for each of the p-values. If None, equal weights are used.
    
    Returns:
        pval: <float>
            The ACAT combined p-value.
    '''
    
    if any(np.isnan(pvalues)):
        raise Exception("Cannot have NAs in the p-values.")
    if any([(i>1)|(i<0) for i in pvalues]):
        raise Exception("P-values must be between 0 and 1.")
    if any([i==1 for i in pvalues])&any([i==0 for i in pvalues]):
        raise Exception("Cannot have both 0 and 1 p-values.")
    if any([i==0 for i in pvalues]):
        print("Warn: p-values are exactly 0.")
        return 0
    if any([i==1 for i in pvalues]):
        print("Warn: p-values are exactly 1.")
        return 1
    if weights==None:
        weights = [1/len(pvalues) for i in pvalues]
    elif len(weights)!=len(pvalues):
        raise Exception("Length of weights and p-values differs.")
    elif any([i<0 for i in weights]):
        raise Exception("All weights must be positive.")
    else:
        weights = [i/len(weights) for i in weights]
    
    pvalues = np.array(pvalues)
    weights = np.array(weights)
    
    if any([i<1e-16 for i in pvalues])==False:
        cct_stat = sum(weights*np.tan((0.5-pvalues)*np.pi))
    else:
        is_small = [i<(1e-16) for i in pvalues]
        is_large = [i>=(1e-16) for i in pvalues]
        cct_stat = sum((weights[is_small]/pvalues[is_small])/np.pi)
        cct_stat += sum(weights[is_large]*np.tan((0.5-pvalues[is_large])*np.pi))
    
    if cct_stat>1e15:
        pval = (1/cct_stat)/np.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)
    
    return pval


# Function for applying ACAT to each row, handling cases with p-values of 0 or 1
def compute_combined_p_value(row):
    """
    Computes a combined p-value using the ACAT (Aggregated Cauchy Association Test) method
    for the adjusted chi-square p-value and permutation p-value.

    Parameters:
        row (pd.Series): A row from a pandas DataFrame containing the following columns:
            - 'chi_p_value_adj': Adjusted p-value from a chi-square test.
            - 'perm_p_value_adj': Adjusted p-value from a permutation test.

    Returns:
        float: The combined p-value. If any p-value is exactly 0, the result is 0.
    """

    # Extract chi-square and permutation p-values from the row
    pvalues = [row['chi_p_value_adj'], row['perm_p_value_adj']]
    
    # Define a small epsilon value to avoid zero p-values
    eps = 1e-300  # Prevent underflow in ACAT test
    
    # Ensure p-values are within a valid range
    pvalues = [max(p, eps) for p in pvalues]  # Avoid p-values of 0
    pvalues = [1 - (1 / len(pvalues)) if p == 1 else p for p in pvalues]  # Adjust p = 1 cases

    # Compute the ACAT combined p-value
    return acat_test(pvalues, weights=[0.9, 0.1])  # weights: chi and perm


# Function for performing permutation tess 
def permutation_test_all_clusters(adata, cluster_info, observed_expression, expected_expression, 
                                  n_permutations=1000, use_zeros=True, random_seed=42, path=None):
    """
    Perform permutation test on all genes for each cluster, using other clusters as null distribution.

    Parameters:
    -----------
    adata : AnnData
        An AnnData object containing the single-cell expression data.
        
    cluster_info : str
        The column name in `adata.obs` that contains cluster information.
    
    observed_expression : pd.DataFrame
        Observed expression values with genes as columns and spots as rows.
        
    expected_expression : pd.DataFrame
        Expected expression values with genes as columns and spots as rows.

    n_permutations : int, optional
        Number of permutations for permutation testing (default is 100).
    
    use_zeros : bool, optional
        Whether to include zero values in the comparison (default is True).
    
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    path : str, optional
        Path to save the QQ plot (default is None).    
        
    Returns:
    --------
    results_df : pandas.DataFrame
        A DataFrame containing the permutation test results for all genes across clusters.
    """
    
    rng = np.random.default_rng(random_seed)
    results = []

    # List of unique clusters
    clusters = adata.obs[cluster_info].cat.categories
    
    for cluster in clusters:
        # Subset observed and expected data for the target cluster
        target_observed = observed_expression.loc[observed_expression[cluster_info] == cluster, :].select_dtypes(include=[np.number])
        target_expected = expected_expression.loc[expected_expression[cluster_info] == cluster, :].select_dtypes(include=[np.number])

        # Subset observed and expected data for all clusters except the target cluster
        other_observed = observed_expression.loc[observed_expression[cluster_info] != cluster, :].select_dtypes(include=[np.number])
        other_expected = expected_expression.loc[expected_expression[cluster_info] != cluster, :].select_dtypes(include=[np.number])
        
        if not use_zeros:
            # Replace 0 with NaN to exclude them from mean calculation
            target_observed_no_zeros = target_observed.replace(0, np.nan)
            target_expected_no_zeros = target_expected.replace(0, np.nan)
            
            # Calculate the difference in mean values for each gene column
            target_diff = target_observed_no_zeros.astype(float).mean() - target_expected_no_zeros.astype(float).mean()
        else:
            # Calculate observed - expected difference for the target cluster
            target_diff = target_observed.astype(float).mean() - target_expected.astype(float).mean()
            
        # Replace dots or hyphens with underscores in adata.var_names
        adata.var_names = adata.var_names.str.replace('-', '_')
        adata.var_names = adata.var_names.str.replace('.', '_')
        
        for gene in tqdm(adata.var_names, desc=f"Permutation Test Progress for Cluster {cluster}"):
            # Get the observed difference for the target cluster
            observed_value = target_diff[gene]
            
            if not use_zeros:
                # Obtain zero_cell_ids
                zero_cell_ids = other_observed[gene][other_observed[gene] == 0].index

                # Extract barcodes from zero_cell_ids
                zero_barcodes = zero_cell_ids.str.split('-').str[0]  # Remove '-1_before' and keep only the barcode
                
                # Filter out rows in other_observed and other_expected with matching barcodes
                filtered_observed = other_observed[~other_observed.index.str.split('-').str[0].isin(zero_barcodes)]
                filtered_expected = other_expected[~other_expected.index.str.split('-').str[0].isin(zero_barcodes)]
                
                # Calculate null_diff
                null_diff = filtered_observed[gene].values - filtered_expected[gene].values

            else:
                # Calculate differences for all spots in other clusters
                null_diff = other_observed[gene].values - other_expected[gene].values
                    
            # Generate null distribution by permutation
            perm_values = []
            for _ in range(n_permutations):
                perm_sample = rng.choice(null_diff, size=len(target_observed), replace=True)
                perm_values.append(np.mean(perm_sample))
            
            # Calculate permutation p-value based on Z-score
            perm_values = np.array(perm_values)
            perm_mean = np.mean(perm_values)  # Calculate the mean of permuted values (null distribution)
            perm_std = np.std(perm_values) # Calculate the standard deviation of permuted values (null distribution)
                  
            if perm_std == 0:  # Handle cases where the std deviation is zero
                perm_p_value = 1e-4
            else:
                z_score = (observed_value - perm_mean) / perm_std  # Compute Z-score for observed value relative to permuted values
                perm_p_value = 1 - norm.cdf(z_score)  # Calculate p-value based on the Z-score       
                   
            # Store the results
            results.append([gene, cluster, float(perm_p_value)])
    
    # Create the results DataFrame
    results_df = pd.DataFrame(results, columns=['gene', 'cluster', 'perm_p_value'])
    
    # Benjamini-Hochberg correction
    results_df = adjust_p_values_bh(df=results_df, p_value_column='perm_p_value', adjusted_column_name='perm_p_value_adj')    
    
    return results_df


# Clean up the celltype names (remove special symbols and space etc), must be done for regression
def clean_column_names(df):
    df.columns = df.columns.str.replace('-', '_')
    df.columns = df.columns.str.replace('.', '_')
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('+', '_') 
    df.columns = df.columns.str.replace('/', '_')
    return df


# Get the trimmed names of cell types
def obtain_clean_celltype_names(adata):
    adata.obs = clean_column_names(adata.obs)
    orig_celltypes = list(adata.uns['mod']['factor_names'])
    celltypes = [i.replace(' ', '_') for i in orig_celltypes]
    celltypes = [i.replace('(', '') for i in celltypes]
    celltypes = [i.replace(')', '') for i in celltypes]
    celltypes = [i.replace('.', '_') for i in celltypes]
    celltypes = [i.replace('-', '_') for i in celltypes]
    celltypes = [i.replace('+', '_') for i in celltypes]
    celltypes = [i.replace('/', '_') for i in celltypes]
    celltypes = [i.replace(':', '_') for i in celltypes]
    return celltypes


# Gene name mapping for different species: mouse
gene_name_mapping_mouse = {
    "Gt(ROSA)26Sor": "ROSA26",
    "Hprt(tm1(cre)Mnn)": "Hprt-cre",
    "Pten(tm1Hwu)": "Pten-flox",
    "Cdh5(PAC)-CreERT2": "Cdh5-CreERT2",
    "Sox2(tm1(cre/ERT2)Hoch)": "Sox2-CreERT2",
    "Tg(CMV-cre)1Cgn": "CMV-Cre",
    "Tg(Myh6-cre)2182Mds": "Myh6-Cre",
}


# Gene name mapping for different species: rat
gene_name_mapping_rat = {
    "Gt(ROSA)26Rat": "ROSA26",
    "Hprt(tm1Llr)": "Hprt-flox",
    "Pten(tm1Llr)": "Pten-flox",
    "Cdh5(PAC)-CreERT2": "Cdh5-CreERT2",
    "Sox2(tm1Llr)": "Sox2-flox",
    "Tg(CMV-cre)1Ljr": "CMV-Cre",
    "Tg(Myh6-cre)1Lfr": "Myh6-Cre",
}


# Simplify gene names
def simplify_gene_names(adata, species):
    """
    Simplifies gene names in adata.var_names based on the specified species.
    
    Parameters:
    - adata: AnnData object containing gene expression data
    - species: String specifying the species ('mouse' or 'rat')

    This function replaces complex gene names with their corresponding simplified names
    based on predefined mappings for each species.
    """
    if species.lower() == "mouse":
        gene_mapping = gene_name_mapping_mouse
    elif species.lower() == "rat":
        gene_mapping = gene_name_mapping_rat
    else:
        return  # If species is not supported, do nothing

    # Replace gene names based on the mapping
    adata.var_names = adata.var_names.to_series().replace(gene_mapping)

