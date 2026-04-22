# -*- coding: utf-8 -*-

"""Carry out Radical search to identify extreme samples in the dataset and give them a single sample score."""

import argparse
import os
from typing import Callable, Optional, List, Tuple, Any
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas._typing as pdt
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm
import pandera.typing as pat


def do_radical_search(
        data: pd.DataFrame,
        design: pd.DataFrame,
        threshold: float,
        control: str|int,
        control_based: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify the samples with extreme feature values either based on the entire dataset or control population.

    :param data: Dataframe containing the gene expression values
    :param design: Dataframe containing the design table for the data
    :param threshold: Threshold for choosing patients that are "extreme" w.r.t. the controls
    :param control: label used for representing the control in the design table of the data
    :param control_based: The scoring is based on the control population instead of entire dataset
    :return: Dataframe containing the Single Sample scores using radical searching
    """
    # Transpose matrix to get the patients as the rows
    if all(s in data.columns for s in design.index[:5]):
        data_transpose = data.transpose()
    else:
        data_transpose = data
    print(f"Transposed data shape: {data_transpose.shape}, supposed to be [sample, gene]")
    # Give each label an integer to represent the labels during classification
    label_mapping = {
        key: val
        for val, key in enumerate(np.unique(design['Target']))
    }

    # Make sure the number of rows of transposed data and design are equal
    assert len(data_transpose) == len(design), 'Data doesnt match the design matrix'

    # Create a dataframe initialized with 0's [patients x features]
    output_df = pd.DataFrame(0, index=data_transpose.index, columns=data_transpose.columns)

    # Values that are greater than the threshold or lesser than negative threshold are considered as extremes.
    upper_thresh = 1 - (threshold / 100)
    lower_thresh = (threshold / 100)

    if control_based:
        controls = data_transpose[list(design.Target == control)]

        # Calculate the empirical cdf for every gene and get the cdf score for the data
        control_ecdf = controls.apply(_get_ecdf, step=False, extrapolate=True).values
        cdf_score = _apply_func(data_transpose, control_ecdf).fillna(0)

        # Check if each patient's feature is over or under expressed compared to the control population
        output_df = pd.DataFrame(np.where(cdf_score.values > upper_thresh, 1, output_df.values))
        output_df = pd.DataFrame(np.where(cdf_score.values < lower_thresh, -1, output_df.values))

    else:
        # Calculate the empirical cdf for every gene and get the cdf score for the data
        feature_to_ecdf = {
            feature: _get_ecdf(data_transpose[feature].values)
            for feature in data_transpose
            if len(data_transpose[feature].unique()) > 1  # Check not all values are the same
        }

        # Iterate over patients and check if any of its features is significant
        for patient_index, features in data_transpose.iterrows():

            # Iterate over patient features
            for feature, value in features.items():

                # Skip if feature has no calculated eCDF
                if feature not in feature_to_ecdf:
                    continue

                # Calculate position of the patient in the distribution of the feature
                patient_position_in_distribution = float(feature_to_ecdf[feature]([value])[0])

                if patient_position_in_distribution <= lower_thresh:
                    output_df.loc[patient_index, feature] = -1

                if patient_position_in_distribution > upper_thresh:
                    output_df.loc[patient_index, feature] = 1

    output_df.columns = data_transpose.columns
    output_df.index = data_transpose.index

    summary_df = output_df.apply(pd.Series.value_counts)

    # Add labels to the data samples
    label = design['Target'].map(label_mapping)
    label.reset_index(drop=True, inplace=True)

    output_df['label'] = label.values

    return output_df, summary_df


def _get_ecdf(
        obs: pdt.ArrayLike,
        side: str = 'right',
        step: bool = True,
        extrapolate: bool = False
) -> Any:
    """Calculate the Empirical CDF of an array and return it as a function.

    :param obs: Observations
    :param side: Defines the shape of the intervals constituting the steps. 'right' correspond to [a, b) intervals
        and 'left' to (a, b]
    :param step: Boolean value to indicate if the returned value must be a step function or an continuous based on
        interpolation or extrapolation function
    :param extrapolate: Boolean value to indicate if the continuous must be based on extrapolation
    :return: Empirical CDF as a function
    """
    if step:
        return ECDF(x=obs, side=side)
    else:
        obs = np.array(obs, copy=True)
        obs.sort()

        num_of_obs = len(obs)

        y = np.linspace(1. / num_of_obs, 1, num_of_obs)

        if extrapolate:
            return interp1d(obs, y, bounds_error=False, fill_value="extrapolate")  # type: ignore
        else:
            return interp1d(obs, y)


def _apply_func(
        df: pd.DataFrame,
        func_list: npt.NDArray[Any]
) -> pd.DataFrame:
    """Apply functions from the list (in order) on the respective column.

    :param df: Data on which the functions need to be applied
    :param func_list: List of functions to be applied
    :return: Dataframe which has been processed
    """
    final_df = pd.DataFrame()

    new_columns = [index for index, _ in enumerate(df.columns)]
    old_columns = list(df.columns)

    df.columns = pd.Index(new_columns)

    for idx, i in enumerate(tqdm(df.columns, desc='Searching for radicals: ')):
        final_df[i] = np.apply_along_axis(func_list[idx], 0, df[i].values)  # type: ignore

    final_df.columns = pd.Index(old_columns)

    return final_df


def do_biological_logfc(
    data: pd.DataFrame,
    design: pd.DataFrame,
    threshold: float = 0.1,
    alpha: float = 0.05,
    control: str|int = 0,
    control_based: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    """
    Identifies 'Radicals' based on the logFC Volcano Plot regions:
    1 (overexpressed):  logFC > threshold  AND  adj.P-value < alpha
    -1 (underexpressed): logFC < -threshold AND  adj.P-value < alpha
    0 : Does not meet both criteria.

    Returns:
        _type_: _description_
    """
    # 1. Safety Check: Is the data log-scaled?
    max_val = np.percentile(data.values, 99)
    
    if max_val > 50:
        warnings.warn(f"Data appears to be raw counts (Max: {max_val:.2f}). Applying log2(x + 1) transformation.")
        # Apply log2 transformation: adding 1 avoids log(0) errors
        working_data = np.log2(data + 1)
        if not isinstance(working_data, pd.DataFrame):
            working_data = pd.DataFrame(working_data, index=data.index, columns=data.columns)
    else:
        working_data = data.copy()
    print(f"Working data shape: {working_data.shape}")
    
    # 2. Align data and design -> data_t[samples, genes]
    if all(s in working_data.columns for s in design.index[:5]):
        data_t = working_data.transpose()
    else:
        data_t = working_data
        working_data = working_data.transpose()
         
    print(f"Transpose data shape: {data_t.shape}, supposed to be [sample * gene]")

    control_idx = design[design['Target'] == control].index.to_list()
    case_idx = design[design['Target'] != control].index.to_list()
    
    # 3. Statistical Testing (Gene by Gene)
    results = []
    for gene in working_data.index:
        control_vals = working_data.loc[gene, control_idx]
        case_vals = working_data.loc[gene, case_idx]
        
        # Calculate Mean LogFC (Case Mean - Control Mean)
        mean_logfc = case_vals.mean() - control_vals.mean()
    
        # Handle zero variance to prevent NaNs
        if control_vals.var() == 0 and case_vals.var() == 0:
            p_val = 1.0
        else:
            # Perform Welch's T-Test (Comparing the two distributions): equal_var=False
            _, p_val = stats.ttest_ind(case_vals, control_vals, equal_var=False, nan_policy='omit')
            if np.isnan(p_val): p_val = 1.0 # type: ignore
        
        results.append({'gene': gene, 'logFC': mean_logfc, 'p_value': p_val})
    
    stats_df = pd.DataFrame(results).set_index('gene')
    
    # 4. Multiple Testing Correction (FDR / Benjamini-Hochberg)
    # This prevents false positives when testing thousands of genes
    stats_df['adj_P_val'] = multipletests(stats_df['p_value'], method='fdr_bh')[1]

    # 5. Scoring the Samples (The 'Volcano' Logic)
    # Initialize output matrix [samples x genes]
    output_df = pd.DataFrame(0, index=data_t.index, columns=working_data.index)
    
    # We only mark a gene as 1 or -1 if the GENE ITSELF is significant overall
    # and the individual sample's expression is extreme.
    for gene in working_data.index:
        gene_stats = stats_df.loc[gene]
        
        if gene_stats['adj_P_val'] < alpha:
            # Calculate sample-specific deviation from control mean
            ctrl_mean = working_data.loc[gene, control_idx].mean()
            sample_deviations = data_t[gene] - ctrl_mean
            
            # Upper-Red Region (Significant Up)
            output_df.loc[sample_deviations > threshold, gene] = 1
            # Upper-Blue Region (Significant Down)
            output_df.loc[sample_deviations < -threshold, gene] = -1

    # 6. Create the Summary with counts (-1, 0, 1)
    # We transpose output_df back to [genes x counts] to match stats_df
    counts = output_df.apply(pd.Series.value_counts).fillna(0).astype(int).T
    
    # Ensure all three columns exist even if some aren't present in the data
    for col in [-1, 0, 1]:
        if col not in counts.columns:
            counts[col] = 0
            
    # Rename columns for clarity
    counts = counts.rename(columns={-1: 'count_neg', 0: 'count_neutral', 1: 'count_pos'})
    
    # Combine stats and counts
    summary_df = pd.concat([stats_df, counts[['count_neg', 'count_neutral', 'count_pos']]], axis=1)
    
    # 7. Metadata and Summary
    label_mapping = {key: val for val, key in enumerate(np.unique(design['Target']))}
    output_df['label'] = design.loc[output_df.index, 'Target'].map(label_mapping)
    
    
    return output_df, summary_df

def do_std(
    data: pd.DataFrame,
    design: pd.DataFrame,
    threshold: float,
    control: str|int,
    control_based=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """  Identifies 'Radicals' based on the Standard Deviation of the Control group.
    1  : Sample_Value > (Control_Mean + threshold * Control_Std)
    -1 : Sample_Value < (Control_Mean - threshold * Control_Std)
    0  : Within the expected variation of Controls

    Params:
        data: Dataframe gene expression values (assumed log-scaled)
        design: Dataframe [samples x info] containing the 'Target' column
        threshold: std multiplier
        control: The string or int label for the control group

    Returns:
        Tuple[pd.Dataframe, pd.DataFrame]: _description_
    """

    # 1. Safety Check: Is the data log-scaled?
    max_val = np.percentile(data.values, 99)
    
    if max_val > 50:
        warnings.warn(f"Data appears to be raw counts (Max: {max_val:.2f}). Applying log2(x + 1) transformation.")
        # Apply log2 transformation: adding 1 avoids log(0) errors
        working_data = np.log2(data + 1)
        if not isinstance(working_data, pd.DataFrame):
            working_data = pd.DataFrame(working_data, index=data.index, columns=data.columns)
    else:
        working_data = data.copy()
    print(f"Working data shape: {working_data.shape}")
    
    # 2. Align data and design -> data_t[samples, genes]
    if all(s in working_data.columns for s in design.index[:5]):
        data_t = working_data.transpose()
    else:
        data_t = working_data 
    print(f"Transpose data shape: {data_t.shape}, supposed to be [sample * gene]")

    # 3. Calculate Reference Stats
    if control_based:
        control_samples = design[design['Target'] == control].index
        valid_refs = control_samples.intersection(data_t.index)
    else:
        # Use all available samples as the reference population
        valid_refs = data_t.index
    
    # Calculate stats across the reference rows
    ref_stats = data_t.loc[valid_refs].agg(['mean', 'std'], axis=0).T
    ref_stats['std'] = np.where(ref_stats['std'] == 0, 1e-6, ref_stats['std'])
    
    # 4. Scoring Logic (Vectorized)
    upper_bound = ref_stats['mean'] + (threshold * ref_stats['std'])
    lower_bound = ref_stats['mean'] - (threshold * ref_stats['std'])

    output_df = pd.DataFrame(0, index=data_t.index, columns=data_t.columns)
    output_df[data_t > upper_bound] = 1
    output_df[data_t < lower_bound] = -1

    # 5. Summary Statistics (Safe Alignment)
    summary_df = ref_stats.copy().rename(columns={'mean': 'control_mean', 'std': 'control_std'})
    
    # Count occurrences of -1, 0, 1 for each gene
    # value_counts() on the columns of output_df
    counts = output_df.apply(lambda x: x.value_counts()).fillna(0).T
    
    # Map the count columns specifically using reindex to avoid NaNs
    summary_df['count_neg'] = counts.reindex(columns=[-1]).iloc[:, 0].fillna(0).astype(int)
    summary_df['count_neutral'] = counts.reindex(columns=[0]).iloc[:, 0].fillna(0).astype(int)
    summary_df['count_pos'] = counts.reindex(columns=[1]).iloc[:, 0].fillna(0).astype(int)

    # 6. Add labels
    label_mapping = {key: val for val, key in enumerate(np.unique(design['Target']))}
    output_df['label'] = design.loc[output_df.index, 'Target'].map(label_mapping)
    
    summary_df_1 = output_df.apply(pd.Series.value_counts)
    
    return output_df, summary_df

def process_and_save(
    data: pd.DataFrame, 
    design: pd.DataFrame, 
    threshold: float, 
    control: str|int,
    do_function,
    output_dir: str,
    method: str = 'logfc',
    **kwards
):
    """
    Wrapper to run the search, show progress, and save results.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Starting {method} analysis...")
    
    # Simple progress bar for the high-level step
    with tqdm(total=2, desc="Overall Progress") as pbar:
        # 1. do sample scoring
        output_df, summary_df = do_function(data, design, threshold, control=control, **kwards)
        
        pbar.update(1)
        
        # 2. Save the files
        out_path = os.path.join(output_dir, f"sample_scoring_{method}.csv")
        sum_path = os.path.join(output_dir, f"scoring_summary_{method}.csv")
        #sum_path_2 = os.path.join(output_dir, f"scoring_summary1_{method}.csv")
        
        output_df.to_csv(out_path)
        summary_df.to_csv(sum_path)
        #summary_df_2.to_csv(sum_path_2)
        pbar.update(1)

    print(f"Done! Files saved to {output_dir}")
    return output_df, summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="../data/GEO/GSE33000_ad_hd/GSE33000_exp_2cls.csv")
    parser.add_argument('--design', type=str, default="../data/GEO/GSE33000_ad_hd/GSE33000_meta_2cls.csv")
    parser.add_argument('--output_dir', type=str, default="../data/GEO/GSE33000_ad_hd/sample_scoring")
    parser.add_argument('--method', type=str, default='ecdf', choices=['ecdf', 'std', 'logfc'])
    parser.add_argument('--control', default='Control', choices=['Control', 0], help="Control labels in design")

    parser.add_argument('--control_based', action='store_true', help="Whether use control group as baseline to score extreme.")
    
    args = parser.parse_args()


    df_exp = pd.read_csv(args.exp, index_col=0)
    print(f"Expression df shape: {df_exp.shape}")
    design = pd.read_csv(args.design, index_col=0)
    print(f"design file shape: {design.shape}")

    method_threshold_map = {
        'ecdf':5,
        'logfc':0.1,
        'std':1.5
    }
    method_function_map = {
        'ecdf':do_radical_search,
        'logfc':do_biological_logfc,
        'std':do_std
    }
    for method in ['ecdf', 'std', 'logfc']:
        df, summary = process_and_save(
            data=df_exp.transpose(), 
            design=design, 
            threshold=method_threshold_map[method], 
            control=args.control,
            do_function=method_function_map[method],
            output_dir=args.output_dir,
            method = method,
            control_based=args.control_based
        )

if __name__=="__main__":
    main()