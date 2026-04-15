# Process GEO dataset -> dfs of expression data and metadata
import argparse
import os
import pandas as pd


def get_platform(platform_path):
    with open(platform_path, 'r') as f:
        lines = f.readlines()
        skip_num = 0
        for i, line in enumerate(lines):
            if line.startswith("!platform_table_begin"):
                skip_num = i + 1
                break
    # Read the file starting from that line
    df_platform = pd.read_csv(platform_path, 
                            sep='\t', 
                            skiprows=skip_num)
    # Clean up: GEO often adds a '!platform_table_end' at the very bottom
    df_platform = df_platform[df_platform['ID'].astype(str).str.contains('!') == False]

    return df_platform

def get_expression_data(exp_path, df_platform):
    
    # 1. Read expression data
    df_expression = pd.read_csv(exp_path, 
                            sep='\t', 
                            comment='!', 
                            index_col=0
                            )
    df_expression = df_expression.rename_axis(None, axis=0)
    # 2. Filter to keep only the ID and Gene Symbol (ORF)
    # Drop rows where ORF is missing 
    mapping_df = df_platform[['ID', 'ORF']].dropna(subset=['ORF'])

    # 3. Create the dictionary
    mapping_dict = dict(zip(mapping_df['ID'].astype(str), mapping_df['ORF']))

    # 4. Apply to Expression DataFrame
    # Ensure expression index is also string type to match
    df_expression.index = df_expression.index.astype(str)
    df_expression.index = df_expression.index.map(mapping_dict)

    # 5. Clean up: Remove probes that didn't map to a gene
    df_expression = df_expression[df_expression.index.notna()]

    # 6. Collapse duplicates (Mean of probes for the same gene)
    df_final = df_expression.groupby(level=0).mean()
    
    return df_final

def get_metadata(file_path):
    info_names = []
    rows = []
    index = []
    # Open the file and read line by line to avoid memory/parser issues
    with open(file_path, 'r') as f:
        for line in f:
            # 1. Extract the Sample Info
            if line.startswith("!Sample"):
                info_name = line.split('\t')[0]
                info_name = '_'.join(info_name.split('_')[1:])
                #print(info_name)
                info_names.append(info_name)
                row = [val.strip().replace('"', '') for val in line.split('\t')[1:]]
                rows.append(row)

    # Now the df_meta contains a lot redundent information  
    df_meta = pd.DataFrame(rows)   
    df_meta.columns = df_meta.iloc[1]
    df_meta = df_meta.T
    df_meta.columns = info_names
    df_meta = df_meta.rename_axis(None, axis=0)

    # Keep only the special information
    target_cols = [c for c in ['characteristics_ch1', 'characteristics_ch2'] if c in df_meta.columns]
    df_metadata = df_meta[target_cols]
    
    return df_metadata

def clean_metadata(df):
    df = df.copy()
    new_names = list(df.columns) # Start with existing names

    for i in range(len(df.columns)):
        series = df.iloc[:, i]
        valid_rows = series.dropna()
        if valid_rows.empty: continue
        
        sample_val = str(valid_rows.iloc[0])
        if ':' in sample_val:
            # 1. Extract name and capitalize
            new_names[i] = sample_val.split(':')[0].strip().capitalize()
            # 2. Clean the data in the column
            df.iloc[:, i] = series.str.split(':').str[-1].str.strip()
            
    # 3. Apply the list of names directly (handles duplicates automatically)
    df.columns = new_names
    # 4. Now drop duplicates
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    
    return df


def data_process(matrix_path, platform_path, output_path):

    # 1. get platform df
    df_platform = get_platform(platform_path=platform_path)

    # 2. get expression data
    df_expression = get_expression_data(exp_path=matrix_path,
                                        df_platform=df_platform)

    # 3. get metadata
    df_meta = get_metadata(file_path=matrix_path)
    # clean metadata -> get samples target
    df_meta_clean = clean_metadata(df=df_meta)
    
    # 4. save metadata & expression file
    geo_name = matrix_path.split('/')[-1].split('_')[0]
    save_exp = os.path.join(output_path, f"{geo_name}_exp.csv")
    df_expression.to_csv(save_exp)
    print(f"Expression file is saved to {save_exp}")

    save_meta = os.path.join(output_path, f"{geo_name}_meta.csv")
    df_meta_clean.to_csv(save_meta)
    print(f"Metadata file is saved to {save_meta}")
    
    return df_expression, df_meta_clean

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

def main():
   
    parser = argparse.ArgumentParser(description="Process GEO dataset: Expression and Metadata")
    parser.add_argument("-m", "--matrix", type=str, default=".", 
                        help="Path to the GEO series_matrix.txt file")
    parser.add_argument('g', "--gds_path", type=str, default=".",
                        help="Path to the GDS .soft.gz file")
    parser.add_argument("-p", "--platform", type=str, default=".", 
                        help="Path to the GEO platform (GPL) .txt file")
    parser.add_argument("-o", "--output", type=str, default=".", 
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"Created directory: {args.output}")

    print(f"--- Processing: {args.matrix} ---")
    
    try:
        # 4. Run the process using the parsed arguments
        df_exp, df_meta = data_process(
            matrix_path=args.matrix, 
            platform_path=args.platform, 
            output_path=args.output
        )
        print("Success! Files saved successfully.")
        print(f"Metadata Shape: {df_meta.shape}")
        print(f"Expression Shape: {df_exp.shape}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()