# Process GEO dataset -> dfs of expression data and metadata
from pathlib import Path
from typing import Union
import pandas as pd


def get_platform(platform_path: str,
                 skip_num: int,
                 output: str) -> pd.DataFrame:
    """
    Read a GEO GPL platform annotation file and extract the platform table.

    The platform file contains probe annotations (e.g., probe ID, gene symbol,
    Entrez ID, gene name). This function reads the annotation table beginning
    at the specified row and removes the GEO footer line.

    Parameters
    ----------
    platform_path : str
        Path to the GEO platform (.txt) file.

    skip_num : int
        Number of rows to skip before the platform annotation table begins.
        This value depends on the specific GPL file.

    output : str
        Filename where the extracted platform DataFrame will be saved.

    Returns
    -------
    pd.DataFrame
        Platform annotation DataFrame containing probe annotations.
    """

    df_platform = pd.read_csv(
        platform_path,
        sep="\t",
        skiprows=skip_num
    )

    # Remove GEO footer
    df_platform = df_platform[
        ~df_platform["ID"].astype(str).str.contains("!")
    ]

    df_platform.to_csv(output + "_platform_annotations.csv", index=False)

    return df_platform


def get_expression_data(exp_path: str,
                        df_platform: pd.DataFrame,
                        platform_cols: list,
                        symbol_colname: str,
                        output: str):
    """
    Convert probe-level GEO expression data into gene-level expression data.

    The function performs the following steps:

    1. Read the GEO Series Matrix expression table.
    2. Create a probe-to-gene mapping using the platform annotation.
    3. Replace probe IDs with gene symbols.
    4. Remove probes without gene annotations.
    5. Collapse duplicated genes by averaging probe values.

    Parameters
    ----------
    exp_path : str
        Path to the GEO Series Matrix file.

    df_platform : pd.DataFrame
        Platform annotation DataFrame obtained from ``get_platform()``.

    platform_cols : list
        Columns to retain from the platform annotation.
        Example:
        ['ID', 'Symbol', 'Synonyms']

    symbol_colname : str
        Column name containing gene symbols.
        Example:
        'Symbol' or 'Symbol/ORF'

    output : str
        Filename for saving the generated DataFrames.

    Returns
    -------
    tuple
        (df_mean, df_all, mapping_df)

        df_mean : pd.DataFrame
            Gene-level expression matrix after averaging duplicate probes.

        df_all : pd.DataFrame
            Probe-mapped expression matrix before duplicate collapsing.

        mapping_df : pd.DataFrame
            Probe-to-gene mapping table.
    """

    # Read expression matrix
    df_expression = pd.read_csv(
        exp_path,
        sep="\t",
        comment="!",
        index_col=0
    )

    df_expression = df_expression.rename_axis(None, axis=0)

    # Build probe-to-gene mapping
    mapping_df = df_platform[platform_cols].dropna(
        subset=[symbol_colname]
    )

    mapping_dict = dict(
        zip(
            mapping_df["ID"].astype(str),
            mapping_df[symbol_colname]
        )
    )

    # Map probe IDs to gene symbols
    df_expression.index = df_expression.index.astype(str)
    df_expression.index = df_expression.index.map(mapping_dict)

    # Remove unmapped probes
    df_all = df_expression[df_expression.index.notna()]

    # Collapse duplicate genes
    df_mean = df_all.groupby(level=0).mean()

    # Save outputs
    mapping_df.to_csv(output + "_cleaned_platform.csv")
    df_all.to_csv(output + "_expression_gene_all.csv")
    df_mean.to_csv(output + "_expression_gene_mean.csv")

    return df_mean, df_all, mapping_df


def get_metadata(file_path: str,
                 ) -> pd.DataFrame:
    """
    Extract sample metadata from a GEO Series Matrix file.

    GEO Series Matrix files contain metadata stored as lines beginning with
    '!Sample'. This function extracts these metadata fields and converts them
    into a sample-by-metadata DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the GEO Series Matrix file.

    Returns
    -------
    pd.DataFrame
        Metadata DataFrame indexed by sample accession.
        The returned DataFrame usually contains columns such as
        'characteristics_ch1' and/or 'characteristics_ch2'.
    """

    info_names = []
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("!Sample"):

                info_name = line.split("\t")[0]
                info_name = "_".join(info_name.split("_")[1:])

                info_names.append(info_name)

                row = [
                    val.strip().replace('"', "")
                    for val in line.split("\t")[1:]
                ]

                rows.append(row)

    df_meta = pd.DataFrame(rows)

    df_meta.columns = df_meta.iloc[1]
    df_meta = df_meta.T

    df_meta.columns = info_names
    df_meta = df_meta.rename_axis(None, axis=0)

    target_cols = [
        c for c in ["characteristics_ch1", "characteristics_ch2"]
        if c in df_meta.columns
    ]

    df_metadata = df_meta[target_cols]

    return df_metadata


def clean_metadata(df: pd.DataFrame,
                   output: str) -> pd.DataFrame:
    """
    Clean GEO sample metadata.

    GEO metadata fields often have the form

        diagnosis: AD
        gender: Female
        age: 72

    This function

    - extracts the metadata field names,
    - removes the key prefixes from the values,
    - removes duplicate metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata DataFrame returned by ``get_metadata()``.

    output : str
        filename for saving the cleaned metadata.

    Returns
    -------
    pd.DataFrame
        Cleaned metadata DataFrame indexed by sample accession.
    """

    df = df.copy()

    new_names = list(df.columns)

    for i in range(len(df.columns)):

        series = df.iloc[:, i]
        valid_rows = series.dropna()

        if valid_rows.empty:
            continue

        sample_val = str(valid_rows.iloc[0])

        if ":" in sample_val:

            new_names[i] = sample_val.split(":")[0].strip().capitalize()

            df.iloc[:, i] = (
                series.str.split(":")
                .str[-1]
                .str.strip()
            )

    df.columns = new_names

    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    df.to_csv(output + "_metadata_clean.csv")

    return df


def process_gds_soft(file_path):
    metadata_rows = []
    expression_start_line = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            # 1. Capture Metadata with improved parsing
            if line.startswith('#GSM'):
                # Split at the first '='
                parts = line.split(' = ', 1)
                gsm_id = parts[0].replace('#', '').strip()
                
                # GEO GDS files often use "Value for GSMxxx: [Info]; src: [Source]"
                # We want the [Info] and [Source] parts
                content = parts[1].split('Value for ' + gsm_id + ':')[-1].strip()
                
                if '; src: ' in content:
                    info, source = content.split('; src: ', 1)
                else:
                    info, source = content, ""
                
                metadata_rows.append({
                    'Sample': gsm_id, 
                    'Group_Info': info.strip(), 
                    'Source_Tissue': source.strip()
                })
            
            # 2. Find the start of the data table
            if line.startswith('!dataset_table_begin'):
                expression_start_line = i + 1
                break

    # 3. Read Expression Data
    df_all = pd.read_csv(file_path, sep='\t', skiprows=expression_start_line)
    
    # Clean up the end of the file
    if str(df_all.iloc[-1, 0]).startswith('!'):
        df_all = df_all.iloc[:-1]

    # 4. Handle Gene Symbol column variations
    # Look for "Gene symbol", "Gene Symbol", or "gene_symbol"
    potential_symbol_cols = ['Gene symbol', 'Gene Symbol', 'gene_symbol', 'IDENTIFIER']
    symbol_col = None
    for col in potential_symbol_cols:
        if col in df_all.columns:
            symbol_col = col
            break
    print(symbol_col)
    # 5. Separate Expression and Map
    gsm_cols = [c for c in df_all.columns if c.startswith('GSM')]
    df_expression = df_all[['ID_REF'] + gsm_cols].copy()
    
    if symbol_col:
        # Create mapping dictionary
        mapping = dict(zip(df_all['ID_REF'], df_all[symbol_col]))
        df_expression['ID_REF'] = df_expression['ID_REF'].map(mapping)
        
        # Clean up: Remove rows where gene symbol is missing or empty
        df_expression = df_expression.dropna(subset=['ID_REF'])
        df_expression = df_expression[df_expression['ID_REF'] != '']
        
        # Set gene symbols as index and convert to numeric
        df_expression.set_index('ID_REF', inplace=True)
        df_expression = df_expression.apply(pd.to_numeric, errors='coerce')
        
        # Collapse duplicates (Mean of probes for the same gene)
        df_expression = df_expression.groupby(level=0).mean()

    # 6. Create Metadata DF
    df_metadata = pd.DataFrame(metadata_rows).set_index('Sample')
    df_metadata.index.name = None
    
    return df_expression, df_metadata



def clean_gds_metadata(df_meta):
    """
    Specific cleaner for the '#GSM' metadata format in GDS files.
    """
    df = df_meta.copy()
    
    # Split "24 years old Male" into separate Age and Gender columns
    # This regex looks for 'years old' to split or common patterns
    if 'Characteristics' in df.columns:
        # Example split logic: "24 years old Male" -> ["24", "Male"]
        extracted = df['Characteristics'].str.extract(r'(\d+)\s*years\s*old\s*(.*)')
        df['Age'] = extracted[0]
        df['Gender'] = extracted[1]
        df.drop(columns=['Characteristics'], inplace=True)
        
    return df

