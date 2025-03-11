import os
import logging
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class DataIngestion:
    def __init__(self, log_level=logging.INFO):
        # Configure logging
        logging.basicConfig(
            level=log_level, # Loging General Info.
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
            
        self.logger = logging.getLogger(__name__) # Create logger object with current file name.
        
    def validate_file_paths(self, file_paths: List[str]) -> List[str]:
        '''Validate and filter existing file paths. 
            Returns list of valid, existing file paths'''
            
        valid_paths = []
        for path in file_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                self.logger.warning(f"File not found: {path}")

        return valid_paths
        
    def load_dataset(self, file_path: str):
        '''Load a signle CSV dataset'''
        try:
            
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            
            self.logger.info(f"Successfully loaded: {file_path}")
            self.logger.info(f"Records: {len(df)}, Columns: {len(df.columns)}")

            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def handle_missing_values(self, df, numeric_strategy='mean', categorical_strategy='most_frequent'):
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            # First, replace infinity values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Check for missing values
            missing_count = df.isnull().sum().sum()
            self.logger.info(f"Found {missing_count} missing values")
            
            if missing_count > 0:
                df_imputed = df.copy()
                
                # Separate numeric and non-numeric columns
                numeric_cols = df_imputed.select_dtypes(include=np.number).columns
                non_numeric_cols = df_imputed.select_dtypes(exclude=np.number).columns
                
                # Handle numeric columns with SimpleImputer
                if len(numeric_cols) > 0:
                    self.imputer_num = SimpleImputer(strategy=numeric_strategy)
                    df_imputed[numeric_cols] = self.imputer_num.fit_transform(df_imputed[numeric_cols])
                    self.logger.info(f"Imputed numeric columns using {numeric_strategy} strategy")
                
                # Handle non-numeric columns with most_frequent strategy
                if len(non_numeric_cols) > 0:
                    self.imputer_cat = SimpleImputer(strategy=categorical_strategy)
                    df_imputed[non_numeric_cols] = self.imputer_cat.fit_transform(df_imputed[non_numeric_cols])
                    self.logger.info(f"Imputed non-numeric columns using {categorical_strategy} strategy")
                
                return df_imputed
            else:
                self.logger.info("No missing values to impute")
                return df
        except Exception as e:
            self.logger.error(f"Error during missing value handling: {str(e)}")
            raise
        
    def consolidate_datasets(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        '''Consolidate multiple network intrusion detection datasets'''
        try:
            # Validate file paths
            valid_paths = self.validate_file_paths(file_paths)
            
            if not valid_paths:
                self.logger.error("No valid dataset files found!")
                return None
            
            # Load datasets
            dataframes = []
            for path in valid_paths:
                df = self.load_dataset(path)
                if df is not None:
                    dataframes.append(df)
            # Consolidate datasets
            if not dataframes:
                self.logger.error("No datasets could be loaded!")
                return None
            
            consolidated_df = pd.concat(dataframes, ignore_index=True)
            
            # Clean Column names - remove white spaces.
            consolidated_df.columns = consolidated_df.columns.str.strip()
            
            self.logger.info(f"Consolidated Dataset:")
            self.logger.info(f"Total Records: {len(consolidated_df)}")
            self.logger.info(f"Total Features: {len(consolidated_df.columns)}")
            
            return consolidated_df
        except Exception as e:
            self.logger.error(f"Consolidation error: {e}")
            return None
        
    def feature_selection(self, file_paths: List[str]) -> Optional[pd.DataFrame]:
        
        try:
            
            consolidated_df = self.consolidate_datasets(file_paths)
            
            # FEATURES SELECTED AFTER ANALYSIS. 
            selected_features = [
                "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
                "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
                "Fwd IAT Mean", "Fwd IAT Std", "FIN Flag Count", "SYN Flag Count",
                "ACK Flag Count", "RST Flag Count", "Fwd Packets/s", "Bwd Packets/s",
                "Packet Length Mean", "Init_Win_bytes_forward", "Subflow Fwd Packets",
                "Active Mean", "Idle Mean"
            ]
        
            df = consolidated_df[selected_features + ["Label"]]
            
            # Handling Missing Values
            df_clean = self.handle_missing_values(df)
            
            # Group Attacks.
            label_mapping = {
                'BENIGN': 'BENIGN',
                'DoS Hulk': 'Network Attacks',
                'DoS GoldenEye': 'Network Attacks',
                'DoS slowloris': 'Network Attacks',
                'DoS Slowhttptest': 'Network Attacks',
                'PortScan': 'Network Attacks',
                'DDoS': 'Network Attacks',
                'FTP-Patator': 'Authentication Attacks',
                'SSH-Patator': 'Authentication Attacks',
                'Bot': 'Other Attacks',
                'Web Attack � Brute Force': 'Web Attack',
                'Web Attack � XSS': 'Web Attack',
                'Web Attack � Sql Injection': 'Web Attack',
                'Infiltration': 'Other Attacks',
                'Heartbleed': 'Other Attacks'
            }
                        
            df_clean['Label'] = df_clean['Label'].map(label_mapping)
            
            self.logger.info(f"Successfully Grouped Similar Labels")
            
            # Partial Undersampling.
            undersampler = RandomUnderSampler(sampling_strategy={'BENIGN': 500000}, random_state=42)
            df_undersampled, _ = undersampler.fit_resample(df_clean, df_clean['Label'])
            
            # Modrate Oversampling.
            X = df_undersampled.drop('Label', axis=1) 
            y = df_undersampled['Label']  
            
            sampling_strategy = {'Web Attack': 5000, 'Other Attacks': 5000}
            
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled['Label'] = y_resampled
                    
            return df_resampled
        
        except Exception as e:
            self.logger.error(f"Feature Selection Error: {e}")
            return None
        
    def save_dataset(self, df, folder="data/processed", filename="dataset.csv"):
        try:
            
            if isinstance(df, pd.DataFrame):
                os.makedirs(folder, exist_ok=True)
                filepath = os.path.join(folder, filename)
                df.to_csv(filepath, index=False)
                
                self.logger.info(f"Dataset saved successfully to {filepath}")
            else:
                self.logger.info("Input is not a DataFrame")
        except Exception as e:
            self.logger.error(f"An error occurred while saving the dataset: {e}")
            

if __name__ == '__main__':
    
    file_paths = [
        'data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        'data/raw/Monday-WorkingHours.pcap_ISCX.csv',
        'data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        'data/raw/Tuesday-WorkingHours.pcap_ISCX.csv',
        'data/raw/Wednesday-workingHours.pcap_ISCX.csv'
    ]

    dataIngestion = DataIngestion()
    df = dataIngestion.feature_selection(file_paths)
    dataIngestion.save_dataset(df)
    
    