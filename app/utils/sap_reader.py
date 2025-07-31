import pandas as pd
from typing import Optional

def read_sap_bi_excel(file_path: str, file_type: str = "unknown") -> pd.DataFrame:
    """Read SAP BI Excel files by trying all sheets and finding the one with actual data"""
    try:
        print(f"Reading SAP BI Excel file: {file_path} (type: {file_type})")
        
        excel_data = pd.read_excel(file_path, sheet_name=None)
        print(f"Found sheets: {list(excel_data.keys())}")
        
        valid_dfs = []
        
        for sheet_name, sheet_df in excel_data.items():
            print(f"Checking sheet '{sheet_name}': shape {sheet_df.shape}")
            
            if sheet_name.startswith('_com.sap.ip.bi.xl.hiddensheet'):
                print(f"  Skipping hidden sheet: {sheet_name}")
                continue
                
            if sheet_name.startswith('__'):
                print(f"  Skipping system sheet: {sheet_name}")
                continue
            
            if not sheet_df.empty and sheet_df.shape[0] > 0 and sheet_df.shape[1] > 0:
                non_null_rows = sheet_df.dropna(how='all').shape[0]
                if non_null_rows > 1:
                    print(f"  Valid data sheet found: {sheet_name} with {non_null_rows} rows")
                    sheet_df['source_file'] = file_path
                    sheet_df['source_sheet'] = sheet_name
                    valid_dfs.append(sheet_df)
                else:
                    print(f"  Sheet {sheet_name} has only headers/metadata")
            else:
                print(f"  Sheet {sheet_name} is empty")
        
        if valid_dfs:
            combined_df = pd.concat(valid_dfs, ignore_index=True)
            print(f"Combined data shape: {combined_df.shape}")
            print(f"Combined data columns: {list(combined_df.columns)}")
            return combined_df
        else:
            print("No valid data sheets found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error reading SAP BI Excel file {file_path}: {e}")
        return pd.DataFrame()

def get_column_info(df: pd.DataFrame) -> dict:
    """Get column information including data types"""
    column_info = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_info[col] = "numeric"
        else:
            column_info[col] = "categorical"
    return column_info
