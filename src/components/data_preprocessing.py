import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DataPreprocessor:
    def __init__(self, log_level=logging.INFO):
        # Configure logging
        logging.basicConfig(
            level=log_level, # Loging General Info.
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
            
        self.logger = logging.getLogger(__name__) 
        
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
    
    def encode_categorical_features(self, df, target_column=None):
        '''Encode categorical features in the DataFrame.'''
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")

            df_encoded = df.copy()
            
            # Encode the target column if provided.
            if target_column and target_column in df_encoded.columns:
                target_encoder = LabelEncoder()
                df_encoded[target_column] = target_encoder.fit_transform(df_encoded[target_column])
                self.logger.info(f"Encoded target column: {target_column}")
                
            # Encode categorialcal column
            categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
            if target_column and target_column in categorical_cols:
                categorical_cols = categorical_cols.drop(target_column)
                
            for col in categorical_cols:
                try:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col])
                    self.logger.info(f"Encoded categorical column: {col}")
                    
                except Exception as col_error:
                    self.logger.warning(f"Could not encode column {col}: {str(col_error)}") 
            
            return df_encoded
        except Exception as e:
            self.logger.error(f"Error during categorical encoding: {str(e)}")
            raise
        
    
    def scale_features(self, df, target_column=None):
        '''Scale features in the Dataframe.'''
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            df_scaled = df.copy()
            
            numeric_cols = df.select_dtypes(include=np.number).columns
            if target_column and target_column in numeric_cols:
                numeric_cols = numeric_cols.drop(target_column)
                
            if len(numeric_cols) > 0:
                scaler = StandardScaler()    
                df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
                self.logger.info(f"Scaled {len(numeric_cols)} numeric columns")

                return df_scaled
            
            else:
                self.logger.info("No numeric columns to scale")
                return df_scaled
            
        except Exception as e:
            self.logger.error(f"Error during feature scaling: {str(e)}")
            raise
        
    def prepare_data_for_modeling(self, df, target_column):
        '''Prepare the data for modeling by performing feature-target split.'''
        
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame")
            
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            self.logger.info(f"Data prepared for modeling: X shape {X.shape}, y shape {y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error during data preparation: {str(e)}")
            raise        
        
    def fit_transform(self, df, target_column, apply_scaling=True, apply_smote=True):
        '''Apply the complete preprocessing pipeline on dataset'''
        try:
            if df is None or df.empty:
                raise ValueError("DataFrame is empty or None")
            
            # Encode Categorical Features.
            df_encoded = self.encode_categorical_features(df, target_column)
            
            # Scale the data.
            df_scaled = self.scale_features(df_encoded, target_column=target_column)
            
            # Handle Imbalance Data.
            X_train, X_test, y_train, y_test = self.prepare_data_for_modeling(df_scaled, target_column)
            
            # Save Separated Features And Target(Label)
            X_train.to_csv("data/processed/X_train.csv", index=False)
            y_train.to_csv("data/processed/y_train.csv", index=False)
            X_test.to_csv("data/processed/X_test.csv", index=False)
            y_test.to_csv("data/processed/y_test.csv", index=False)
            
            self.logger.info(f"Saved Preprocessed Data X_train, X_test, y_train, y_test")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            self.logger.error(f"Error fit transform: {str(e)}")
            raise
        
if __name__ == '__main__':

    df = pd.read_csv('data\processed\dataset.csv')
    data_preprocessor = DataPreprocessor()
    print(df.head())
    
    X_train, X_test, y_train, y_test = data_preprocessor.fit_transform(df, 'Label')
    print(X_train.head())
    print(y_train.head())
    
    
    
    
    
    