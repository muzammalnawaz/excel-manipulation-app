from typing import List, Dict
import pandas as pd
import numpy as np

from .base_processor import BaseProcessor
from ...core.exceptions import FileProcessingError

class DataSummarizerProcessor(BaseProcessor):
    """Processor for summarizing data with groupby operations"""
    
    VALID_OPERATIONS = ['sum', 'mean', 'median', 'max', 'min', 'count']
    
    def validate_inputs(self, file_path: str, groupby_columns: List[str], 
                       numeric_columns: List[str], operation: str, available_columns: list) -> bool:
        """Validate summarizer inputs"""
        if not groupby_columns:
            raise FileProcessingError("At least one groupby column is required")
        
        if not numeric_columns:
            raise FileProcessingError("At least one numeric column is required")
        
        if operation not in self.VALID_OPERATIONS:
            raise FileProcessingError(f"Operation '{operation}' is not valid. Choose from: {', '.join(self.VALID_OPERATIONS)}")
        
        # Check if columns exist
        for col in groupby_columns:
            if col not in available_columns:
                raise FileProcessingError(f"Groupby column '{col}' not found in the file")
        
        for col in numeric_columns:
            if col not in available_columns:
                raise FileProcessingError(f"Numeric column '{col}' not found in the file")
        
        return True
    
    def process(self, file_path: str, groupby_columns: List[str], 
               numeric_columns: List[str], operation: str, available_columns: list) -> str:
        """Summarize data by grouping and applying aggregation operations to multiple numeric columns"""
        try:
            # Validate inputs
            self.validate_inputs(file_path, groupby_columns, numeric_columns, operation, available_columns)
            
            # Read the Excel file
            df = self.read_excel_file(file_path)
            df = self.clean_dataframe(df)
            
            # Validate columns exist in dataframe
            for col in groupby_columns:
                if col not in df.columns:
                    raise FileProcessingError(f"Groupby column '{col}' not found in the file")
            
            for col in numeric_columns:
                if col not in df.columns:
                    raise FileProcessingError(f"Numeric column '{col}' not found in the file")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise FileProcessingError(f"Column '{col}' is not numeric")
            
            # Clean data by removing rows with missing values in key columns
            df_clean = df.dropna(subset=groupby_columns + numeric_columns)
            
            if df_clean.empty:
                raise FileProcessingError("No valid data remaining after removing missing values")
            
            # Perform groupby operation
            grouped = df_clean.groupby(groupby_columns)
            
            # Apply the specified operation
            if operation == 'sum':
                result = grouped[numeric_columns].sum().reset_index()
            elif operation == 'mean':
                result = grouped[numeric_columns].mean().reset_index()
            elif operation == 'median':
                result = grouped[numeric_columns].median().reset_index()
            elif operation == 'max':
                result = grouped[numeric_columns].max().reset_index()
            elif operation == 'min':
                result = grouped[numeric_columns].min().reset_index()
            elif operation == 'count':
                result = grouped[numeric_columns].count().reset_index()
            
            # Rename columns to include operation
            rename_dict = {}
            for col in numeric_columns:
                rename_dict[col] = f"{col}_{operation}"
            result = result.rename(columns=rename_dict)
            
            # Add total row for sum, mean, and count operations
            if operation in ['sum', 'mean', 'count']:
                total_row = {}
                for col in groupby_columns:
                    total_row[col] = 'TOTAL'
                
                for col in numeric_columns:
                    result_col = f"{col}_{operation}"
                    if operation == 'sum':
                        total_row[result_col] = result[result_col].sum()
                    elif operation == 'mean':
                        total_row[result_col] = result[result_col].mean()
                    elif operation == 'count':
                        total_row[result_col] = result[result_col].sum()
                
                result = pd.concat([result, pd.DataFrame([total_row])], ignore_index=True)
            
            # Generate output path and save
            output_path = self.get_output_path("summarized")
            sheet_data = {"Summary": result}
            self.save_excel_with_formatting(sheet_data, output_path)
            
            print(f"Data summarized successfully: {len(result)-1 if operation in ['sum', 'mean', 'count'] else len(result)} groups, {len(numeric_columns)} numeric columns, operation: {operation}")
            return output_path
            
        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise e
            raise FileProcessingError(f"Error summarizing data: {str(e)}")