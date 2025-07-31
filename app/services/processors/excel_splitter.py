from typing import Dict
import pandas as pd

from .base_processor import BaseProcessor
from ...core.exceptions import FileProcessingError

class ExcelSplitterProcessor(BaseProcessor):
    """Processor for splitting Excel files by column values"""
    
    def validate_inputs(self, file_path: str, column_name: str, available_columns: list) -> bool:
        """Validate that column exists in the file"""
        if not column_name:
            raise FileProcessingError("Column name is required")
        
        if column_name not in available_columns:
            raise FileProcessingError(f"Column '{column_name}' not found in the file")
        
        return True
    
    def process(self, file_path: str, column_name: str, available_columns: list) -> str:
        """Split Excel file by column values into multiple sheets"""
        try:
            # Validate inputs
            self.validate_inputs(file_path, column_name, available_columns)
            
            # Read the Excel file
            df = self.read_excel_file(file_path)
            df = self.clean_dataframe(df)
            
            # Validate column exists and has data
            if column_name not in df.columns:
                raise FileProcessingError(f"Column '{column_name}' not found in the file")
            
            if df[column_name].isna().all():
                raise FileProcessingError(f"Column '{column_name}' contains no data")
            
            # Group data by column values
            grouped_data = df.groupby(column_name)
            
            # Prepare data for Excel sheets
            sheet_data = {}
            sheet_counter = 1
            
            for group_name, group_data in grouped_data:
                sheet_name = self.generate_safe_sheet_name(str(group_name))
                
                # Handle duplicate sheet names
                original_sheet_name = sheet_name
                counter = 1
                while sheet_name in sheet_data:
                    sheet_name = f"{original_sheet_name}_{counter}"
                    counter += 1
                
                # If sheet name is still empty after cleaning, use a default
                if not sheet_name.strip():
                    sheet_name = f"Sheet_{sheet_counter}"
                    sheet_counter += 1
                
                sheet_data[sheet_name] = group_data
            
            # Generate output path and save
            output_path = self.get_output_path("processed")
            self.save_excel_with_formatting(sheet_data, output_path)
            
            print(f"Excel file split into {len(sheet_data)} sheets successfully")
            return output_path
            
        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise e
            raise FileProcessingError(f"Error splitting Excel file: {str(e)}")