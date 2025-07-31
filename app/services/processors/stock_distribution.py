from typing import Dict, List
import pandas as pd
import numpy as np
from collections import defaultdict

from .base_processor import BaseProcessor
from ...core.exceptions import FileProcessingError

class StockDistributionProcessor(BaseProcessor):
    """Processor for stock distribution analysis based on SAP BI warehouse and sales data"""
    
    def validate_inputs(self, warehouse_path: str, sales_path: str) -> bool:
        """Validate stock distribution inputs"""
        if not warehouse_path or not sales_path:
            raise FileProcessingError("Both warehouse and sales file paths are required")
        return True
    
    def rename_columns(self, df: pd.DataFrame, column_mapping: Dict[int, str]) -> pd.DataFrame:
        """Rename columns based on position mapping"""
        columns = list(df.columns)
        new_columns = {}
        
        for position, new_name in column_mapping.items():
            if position < len(columns):
                new_columns[columns[position]] = new_name
        
        # Keep remaining columns as-is
        for i, col in enumerate(columns):
            if i not in column_mapping:
                new_columns[col] = col
        
        return df.rename(columns=new_columns)
    
    def get_warehouse_column_mapping(self) -> Dict[int, str]:
        """Get standard warehouse column mapping"""
        return {
            0: 'Company Code',
            1: 'Company Code Name',
            2: 'Hierarchy Level 2 Code',
            3: 'Department Name',
            4: 'Hierarchy Level 1 Code', 
            5: 'Hierarchy Level 1 Description',
            6: 'Nesto MC Code',
            7: 'Nesto MC Description',
            8: 'Brand Code',
            9: 'Brand Description',
            10: 'Article Code',
            11: 'Article Description'
        }
    
    def get_sales_column_mapping(self) -> Dict[int, str]:
        """Get standard sales column mapping"""
        return {
            0: 'Company Code',
            1: 'Company Code Name',
            2: 'Hierarchy Level 2 Code',
            3: 'Department Name',
            4: 'Hierarchy Level 1 Code',
            5: 'Hierarchy Level 1 Description', 
            6: 'Nesto MC Code',
            7: 'Nesto MC Description',
            8: 'Brand Code',
            9: 'Brand Description',
            10: 'Article Code',
            11: 'Article Description'
        }
    
    def find_quantity_column(self, df: pd.DataFrame, column_type: str = "warehouse") -> str:
        """Find the appropriate quantity column in the dataframe"""
        for col in df.columns:
            col_str = str(col).lower()
            if column_type == "warehouse":
                if 'total' in col_str and 'quantity' in col_str:
                    return col
            elif column_type == "sales_stock":
                if 'stock' in col_str and 'qty' in col_str and 'current' in col_str:
                    return col
            elif column_type == "sales_qty":
                if 'sales' in col_str and 'qty' in col_str:
                    return col
        return None
    
    def process(self, warehouse_path: str, sales_path: str) -> str:
        """Process stock distribution based on SAP BI warehouse and sales data"""
        try:
            # Validate inputs
            self.validate_inputs(warehouse_path, sales_path)
            
            print("Reading warehouse data from SAP BI file...")
            warehouse_df = self.read_excel_file(warehouse_path, "warehouse")
            
            print("Reading sales data from SAP BI file...")
            sales_df = self.read_excel_file(sales_path, "sales")
            
            if warehouse_df.empty:
                raise FileProcessingError("No valid warehouse data found")
            
            if sales_df.empty:
                raise FileProcessingError("No valid sales data found")
            
            print(f"Data loaded - Warehouse: {warehouse_df.shape}, Sales: {sales_df.shape}")
            
            # Clean dataframes
            warehouse_df = self.clean_dataframe(warehouse_df)
            sales_df = self.clean_dataframe(sales_df)
            
            # Rename columns based on standard SAP BI structure
            warehouse_renamed = self.rename_columns(warehouse_df, self.get_warehouse_column_mapping())
            sales_renamed = self.rename_columns(sales_df, self.get_sales_column_mapping())
            
            print(f"Warehouse columns after renaming: {list(warehouse_renamed.columns)}")
            print(f"Sales columns after renaming: {list(sales_renamed.columns)}")
            
            # Clean data by removing rows with null Article Description
            warehouse_clean = warehouse_renamed.dropna(subset=['Article Description'])
            sales_clean = sales_renamed.dropna(subset=['Article Description'])
            
            print(f"After removing null Article Description - Warehouse: {warehouse_clean.shape}, Sales: {sales_clean.shape}")
            
            # Process warehouse data
            warehouse_processed = self._process_warehouse_data(warehouse_clean)
            
            # Process sales data
            sales_processed = self._process_sales_data(sales_clean)
            
            # Calculate distributions
            distribution_results = self._calculate_distributions(warehouse_processed, sales_processed)
            
            # Create output data
            output_data = self._create_output_data(distribution_results)
            
            # Generate output path and save
            output_path = self.get_output_path("stock_distribution")
            self.save_excel_with_formatting(output_data, output_path)
            
            print(f"Stock distribution analysis completed successfully")
            return output_path
            
        except Exception as e:
            if isinstance(e, FileProcessingError):
                raise e
            raise FileProcessingError(f"Error processing stock distribution: {str(e)}")
    
    def _process_warehouse_data(self, warehouse_df: pd.DataFrame) -> pd.DataFrame:
        """Process warehouse data into standardized format"""
        warehouse_processed = pd.DataFrame()
        warehouse_processed['company_code'] = warehouse_df['Company Code']
        warehouse_processed['department'] = warehouse_df['Department Name']
        warehouse_processed['hierarchy_level1'] = warehouse_df['Hierarchy Level 1 Description']
        warehouse_processed['nesto_mc'] = warehouse_df['Nesto MC Description']
        warehouse_processed['brand'] = warehouse_df['Brand Description']
        warehouse_processed['article_code'] = warehouse_df['Article Code']
        warehouse_processed['article_description'] = warehouse_df['Article Description']
        warehouse_processed['article_key'] = warehouse_df['Article Code'].astype(str)
        
        # Find and add quantity column
        qty_col = self.find_quantity_column(warehouse_df, "warehouse")
        if qty_col:
            warehouse_processed['total_quantity'] = pd.to_numeric(warehouse_df[qty_col], errors='coerce').fillna(0)
            print(f"Found warehouse quantity column: {qty_col}")
        else:
            raise FileProcessingError("Could not find total quantity column in warehouse data")
        
        # Filter valid records
        warehouse_processed = warehouse_processed[
            (warehouse_processed['article_code'].notna()) & 
            (warehouse_processed['article_code'] != '') & 
            (warehouse_processed['total_quantity'] > 0)
        ]
        
        # Handle duplicates by aggregating
        duplicate_articles = warehouse_processed['article_key'].duplicated().sum()
        if duplicate_articles > 0:
            print(f"Found {duplicate_articles} duplicate articles in warehouse data")
            warehouse_processed = warehouse_processed.groupby('article_key').agg({
                'company_code': 'first',
                'department': 'first',
                'hierarchy_level1': 'first', 
                'nesto_mc': 'first',
                'brand': 'first',
                'article_code': 'first',
                'article_description': 'first',
                'total_quantity': 'sum'
            }).reset_index()
        
        return warehouse_processed
    
    def _process_sales_data(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Process sales data into standardized format"""
        sales_processed = pd.DataFrame()
        sales_processed['company_code'] = sales_df['Company Code']
        sales_processed['department'] = sales_df['Department Name']
        sales_processed['category'] = sales_df['Hierarchy Level 1 Description']
        sales_processed['nesto_mc'] = sales_df['Nesto MC Description']
        sales_processed['brand'] = sales_df['Brand Description']
        sales_processed['article_code'] = sales_df['Article Code']
        sales_processed['article_description'] = sales_df['Article Description']
        sales_processed['article_key'] = sales_df['Article Code'].astype(str)
        
        # Find and add stock/sales columns
        stock_col = self.find_quantity_column(sales_df, "sales_stock")
        sales_col = self.find_quantity_column(sales_df, "sales_qty")
        
        if stock_col:
            sales_processed['current_stock'] = pd.to_numeric(sales_df[stock_col], errors='coerce').fillna(0)
            print(f"Found sales stock column: {stock_col}")
        else:
            sales_processed['current_stock'] = 0
            print("Current stock column not found, defaulting to 0")
        
        if sales_col:
            sales_processed['sales_qty'] = pd.to_numeric(sales_df[sales_col], errors='coerce').fillna(0)
            print(f"Found sales quantity column: {sales_col}")
        else:
            sales_processed['sales_qty'] = 0
            print("Sales quantity column not found, defaulting to 0")
        
        # Filter valid records
        sales_processed = sales_processed[
            (sales_processed['article_code'].notna()) & 
            (sales_processed['article_code'] != '')
        ]
        
        # Calculate conversion metrics
        sales_processed['total_grn'] = sales_processed['sales_qty'] + sales_processed['current_stock']
        sales_processed['sales_conversion'] = np.where(
            sales_processed['total_grn'] > 0,
            (sales_processed['sales_qty'] / sales_processed['total_grn']) * 100,
            0
        )
        
        return sales_processed
    
    def _calculate_distributions(self, warehouse_df: pd.DataFrame, sales_df: pd.DataFrame) -> List[Dict]:
        """Calculate stock distributions based on conversion rates"""
        # Find high conversion items (>=80%)
        high_conversion_items = sales_df[sales_df['sales_conversion'] >= 80].copy()
        print(f"High conversion items found: {len(high_conversion_items)}")
        
        # Calculate MC-level conversions
        mc_conversions = {}
        for (company_code, nesto_mc), group in sales_df.groupby(['company_code', 'nesto_mc']):
            total_sales = group['sales_qty'].sum()
            total_grn = group['total_grn'].sum()
            mc_conversion = (total_sales / total_grn * 100) if total_grn > 0 else 0
            mc_conversions[f"{company_code}_{nesto_mc}"] = mc_conversion
        
        # Create warehouse lookup
        warehouse_lookup = warehouse_df.set_index('article_key').to_dict('index')
        print(f"Warehouse articles available: {len(warehouse_lookup)}")
        
        # Distribution logic
        distribution_results = []
        
        for _, item in high_conversion_items.iterrows():
            article_key = item['article_key']
            
            if article_key not in warehouse_lookup:
                continue
                
            warehouse_stock = warehouse_lookup[article_key]
            
            mc_key = f"{item['company_code']}_{item['nesto_mc']}"
            mc_conversion = mc_conversions.get(mc_key, 0)
            
            if mc_conversion > 60:
                max_distribution = min(12, warehouse_stock['total_quantity'])
                
                distribution_results.append({
                    'warehouse_code': warehouse_stock['company_code'],
                    'article_code': warehouse_stock['article_code'],
                    'article_description': warehouse_stock['article_description'],
                    'nesto_mc': item['nesto_mc'],
                    'brand': item['brand'],
                    'site_code': item['company_code'],
                    'article_sales_conversion': round(item['sales_conversion'], 2),
                    'mc_conversion': round(mc_conversion, 2),
                    'warehouse_total_qty': warehouse_stock['total_quantity'],
                    'distribution_qty': max_distribution,
                    'current_site_stock': item['current_stock']
                })
        
        # Handle conflicts - if multiple sites want same article, prioritize by conversion
        final_distribution = self._resolve_conflicts(distribution_results)
        
        print(f"Final distribution count: {len(final_distribution)}")
        return final_distribution
    
    def _resolve_conflicts(self, distribution_results: List[Dict]) -> List[Dict]:
        """Resolve conflicts when multiple sites want the same article"""
        distribution_by_article = defaultdict(list)
        for dist in distribution_results:
            distribution_by_article[dist['article_code']].append(dist)
        
        final_distribution = []
        for article_code, sites in distribution_by_article.items():
            if len(sites) == 1:
                final_distribution.append(sites[0])
            else:
                # Sort by sales conversion (highest first)
                sites.sort(key=lambda x: x['article_sales_conversion'], reverse=True)
                
                warehouse_qty = sites[0]['warehouse_total_qty']
                remaining_qty = warehouse_qty
                
                for site in sites:
                    if remaining_qty <= 0:
                        break
                    
                    allocated_qty = min(site['distribution_qty'], remaining_qty)
                    if allocated_qty > 0:
                        site['distribution_qty'] = allocated_qty
                        final_distribution.append(site)
                        remaining_qty -= allocated_qty
        
        return final_distribution
    
    def _create_output_data(self, distribution_results: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Create output data with pivot table, detailed data, and summary"""
        # Create pivot table structure
        pivot_data = []
        distribution_grouped = defaultdict(list)
        
        for item in distribution_results:
            key = f"{item['warehouse_code']}_{item['article_code']}"
            distribution_grouped[key].append(item)
        
        for key, distributions in distribution_grouped.items():
            warehouse_code, article_code = key.split('_', 1)
            
            pivot_row = {
                'Warehouse Code': warehouse_code,
                'Warehouse Total Quantity': distributions[0]['warehouse_total_qty'],
                'Article Code': distributions[0]['article_code'],
                'Article Description': distributions[0]['article_description'],
                'Nesto MC': distributions[0]['nesto_mc'],
                'Brand': distributions[0]['brand']
            }
            
            for dist in distributions:
                pivot_row[dist['site_code']] = dist['distribution_qty']
            
            pivot_data.append(pivot_row)
        
        # Create detailed data
        detailed_data = [{
            'Warehouse Code': item['warehouse_code'],
            'Article Code': item['article_code'],
            'Article Description': item['article_description'],
            'Nesto MC': item['nesto_mc'],
            'Brand': item['brand'],
            'Site Code': item['site_code'],
            'Warehouse Total Qty': item['warehouse_total_qty'],
            'Distribution Qty': item['distribution_qty'],
            'Article Sales Conversion %': item['article_sales_conversion'],
            'MC Conversion %': item['mc_conversion'],
            'Current Site Stock': item['current_site_stock']
        } for item in distribution_results]
        
        # Create summary statistics
        summary_stats = {
            'Total Distributions': len(distribution_results),
            'Unique Articles': len(set(item['article_code'] for item in distribution_results)),
            'Unique Sites': len(set(item['site_code'] for item in distribution_results)),
            'Total Qty Distributed': sum(item['distribution_qty'] for item in distribution_results),
            'Average Sales Conversion %': round(sum(item['article_sales_conversion'] for item in distribution_results) / len(distribution_results), 2) if distribution_results else 0,
            'Average MC Conversion %': round(sum(item['mc_conversion'] for item in distribution_results) / len(distribution_results), 2) if distribution_results else 0
        }
        
        # Convert to DataFrames
        output_data = {}
        
        if pivot_data:
            output_data['Stock Distribution Pivot'] = pd.DataFrame(pivot_data)
        
        if detailed_data:
            output_data['Detailed Distribution'] = pd.DataFrame(detailed_data)
        
        output_data['Summary'] = pd.DataFrame([summary_stats])
        
        return output_data