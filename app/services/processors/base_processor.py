from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from ...utils.sap_reader import read_sap_bi_excel
from ...core.exceptions import FileProcessingError

class BaseProcessor(ABC):
    """Base class for all file processors"""
    
    def __init__(self, file_id: str):
        self.file_id = file_id
    
    @abstractmethod
    def process(self, *args, **kwargs) -> str:
        """Process files and return output path"""
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate processor inputs"""
        pass
    
    def read_excel_file(self, file_path: str, file_type: str = "unknown") -> pd.DataFrame:
        """Common Excel reading logic using SAP BI reader"""
        try:
            df = read_sap_bi_excel(file_path, file_type)
            if df.empty:
                raise FileProcessingError(f"No valid data found in {file_path}")
            return df
        except Exception as e:
            raise FileProcessingError(f"Error reading Excel file {file_path}: {str(e)}")
    
    def get_output_path(self, suffix: str) -> str:
        """Generate output file path"""
        return f"downloads/{self.file_id}_{suffix}.xlsx"
    
    def clean_dataframe(self, df: pd.DataFrame, required_columns: list = None) -> pd.DataFrame:
        """Clean dataframe by removing tracking columns and validating required columns"""
        # Remove tracking columns if they exist
        for col in ['source_file', 'source_sheet']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Validate required columns exist
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise FileProcessingError(f"Required columns missing: {missing_cols}")
        
        return df
    
    def save_excel_with_formatting(self, data: Dict[str, pd.DataFrame], output_path: str) -> None:
        """Save Excel file with multiple sheets and auto-fit columns"""
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for sheet_name, df in data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-fit columns for all sheets
                workbook = writer.book
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
        
        except Exception as e:
            raise FileProcessingError(f"Error saving Excel file to {output_path}: {str(e)}")
    
    def generate_safe_sheet_name(self, name: str, max_length: int = 31) -> str:
        """Generate a safe Excel sheet name"""
        # Convert to string and truncate
        sheet_name = str(name)[:max_length]
        
        # Replace invalid characters
        invalid_chars = ['/', '\\', '*', '?', '[', ']', ':']
        for char in invalid_chars:
            sheet_name = sheet_name.replace(char, '_')
        
        # Ensure not empty
        if not sheet_name.strip():
            sheet_name = "Sheet_1"
        
        return sheet_name