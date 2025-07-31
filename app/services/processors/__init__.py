from .base_processor import BaseProcessor
from .excel_splitter import ExcelSplitterProcessor
from .data_summarizer import DataSummarizerProcessor
from .stock_distribution import StockDistributionProcessor

class ProcessorFactory:
    """Factory class for creating processors"""
    
    _processors = {
        "split_excel_by_column": ExcelSplitterProcessor,
        "summarize_data": DataSummarizerProcessor,
        "process_stock_distribution": StockDistributionProcessor,
    }
    
    @classmethod
    def create_processor(cls, processor_name: str, file_id: str) -> BaseProcessor:
        """Create a processor instance"""
        if processor_name not in cls._processors:
            raise ValueError(f"Unknown processor: {processor_name}")
        
        processor_class = cls._processors[processor_name]
        return processor_class(file_id)
    
    @classmethod
    def get_available_processors(cls) -> list:
        """Get list of available processor names"""
        return list(cls._processors.keys())

__all__ = [
    'BaseProcessor',
    'ExcelSplitterProcessor', 
    'DataSummarizerProcessor',
    'StockDistributionProcessor',
    'ProcessorFactory'
]