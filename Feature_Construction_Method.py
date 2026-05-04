# ==========================================================================
#  A Hybridized Feature Construction Method Based on Symbolic Regression
#               and Evolutionary Forest For Regression
#
#  Authors: S. GUZEY and E. HANCER
# ==========================================================================
"""
This module implements a hybridized feature construction method that combines:
- Symbolic Regression (via genetic programming)
- Evolutionary Forest techniques
for regression tasks.
"""

import gc
import logging
import warnings
import traceback
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                               AdaBoostRegressor, GradientBoostingRegressor)
from gplearn.genetic import SymbolicTransformer
from evolutionary_forest.forest import EvolutionaryForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Suppress numpy warnings
np.seterr(divide='ignore', invalid='ignore')

# ── GPLearn compatibility patch (sklearn 1.6+) ──────────────────
try:
    import gplearn.genetic as _gp
    from sklearn.utils.validation import check_array as _check_array

    def _gplearn_validate_data(self, X, y=None, **kwargs):
        X = _check_array(X, dtype='numeric')
        self.n_features_in_ = X.shape[1]
        if y is not None:
            y = _check_array(y, ensure_2d=False, dtype='numeric')
            return X, y
        return X

    _gp.BaseSymbolic._validate_data = _gplearn_validate_data
except (ImportError, AttributeError) as e:
    logger.warning(f"GPLearn compatibility patch failed: {e}")

# ── Evolutionary Forest compatibility patch ───────────────────────────
try:
    import evolutionary_forest.forest as _ef_mod
    _ef_mod.consistency_check = lambda learner: None
except (ImportError, AttributeError) as e:
    logger.warning(f"Evolutionary Forest compatibility patch failed: {e}")

def evaluate_regressor_performance(X_train: np.ndarray, y_train: np.ndarray, 
                                   X_test: np.ndarray, y_test: np.ndarray, 
                                   X_train_enhanced: np.ndarray, X_test_enhanced: np.ndarray, 
                                   dataset_name: str) -> list:
    """
    Evaluate and compare regression model performance on original and enhanced features.
    
    Parameters:
    -----------
    X_train, y_train : Original training data
    X_test, y_test : Original test data
    X_train_enhanced, X_test_enhanced : Enhanced training/test data with constructed features
    dataset_name : Name of the dataset
    
    Returns:
    --------
    list : List of performance metrics for each algorithm
    """
    regressor_list = ['RandomForest', 'ExtraTrees', 'AdaBoost', 'GBDT', 'XGBoost', 'LightGBM', 'CatBoost']
    
    regressor_dict = {
        'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'GBDT': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_jobs=1, n_estimators=100, verbosity=0, random_state=42),
        'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=100, verbose=-1, random_state=42),
        'CatBoost': CatBoostRegressor(n_estimators=100, thread_count=1, verbose=False, 
                                       allow_writing_files=False, random_state=42),
    }

    results = []

    for regr_name in regressor_list:
        try:
            logger.info(f"Evaluating {regr_name} on {dataset_name}...")
            
            # Base Model (Original Features)
            base_model = clone(regressor_dict[regr_name])
            base_model.fit(X_train, y_train)
            y_pred_base = base_model.predict(X_test)
            
            base_r2 = r2_score(y_test, y_pred_base)
            base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
            base_mae = mean_absolute_error(y_test, y_pred_base)

            # Enhanced Model (Constructed Features)
            enh_model = clone(regressor_dict[regr_name])
            enh_model.fit(X_train_enhanced, y_train)
            y_pred_enh = enh_model.predict(X_test_enhanced)
            
            enh_r2 = r2_score(y_test, y_pred_enh)
            enh_rmse = np.sqrt(mean_squared_error(y_test, y_pred_enh))
            enh_mae = mean_absolute_error(y_test, y_pred_enh)

            r2_improvement = enh_r2 - base_r2
            rmse_improvement = base_rmse - enh_rmse

            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Base_R2': base_r2,
                'Enhanced_R2': enh_r2,
                'R2_Improvement': r2_improvement,
                'Base_RMSE': base_rmse,
                'Enhanced_RMSE': enh_rmse,
                'RMSE_Improvement': rmse_improvement,
                'Base_MAE': base_mae,
                'Enhanced_MAE': enh_mae,
                'MAE_Improvement': base_mae - enh_mae
            })

        except Exception as e:
            logger.error(f"Error ({regr_name} - {dataset_name}): {str(e)}")
            traceback.print_exc()
            results.append({
                'Algorithm': regr_name,
                'Dataset': dataset_name,
                'Base_R2': 0, 'Enhanced_R2': 0, 'R2_Improvement': 0,
                'Base_RMSE': 0, 'Enhanced_RMSE': 0, 'RMSE_Improvement': 0,
                'Base_MAE': 0, 'Enhanced_MAE': 0, 'MAE_Improvement': 0
            })

    return results



def symbolic_regression_evolutionary_forest_feature_engineering(X: pd.DataFrame, 
                                                                y: pd.Series, 
                                                                dataset_name: str = "CustomDataset",
                                                                n_best_features: int = 10,
                                                                test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hybridized Feature Construction Method Based on Symbolic Regression and Evolutionary Forest.
    
    This method combines Symbolic Regression (via genetic programming) and Evolutionary Forest 
    techniques to construct new features that enhance regression model performance.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    dataset_name : str
        Name of the dataset for reporting
    n_best_features : int
        Number of best features to select from evolutionary forest output
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        - stats_df: Summary statistics of model performance
        - df_raw: Detailed results for all algorithms
    """
    logger.info(f"Starting feature engineering on {dataset_name}...")
    
    all_results = []
    np.seterr(divide='ignore', invalid='ignore')

    try:
        # Data Preparation
        logger.info("Preparing data...")
        X_vals = np.nan_to_num(X.values.astype(np.float64)) if isinstance(X, pd.DataFrame) else np.nan_to_num(X.astype(np.float64))
        y_vals = y.values if isinstance(y, pd.Series) else y
        y_vals = y_vals.astype(np.float64)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_vals, y_vals, test_size=test_size, random_state=42
        )

        # Standardization
        logger.info("Standardizing features...")
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # Stage 1: Symbolic Regression via Genetic Programming
        logger.info("Stage 1: Applying Symbolic Transformer (STGP)...")
        try:
            stgp = SymbolicTransformer(n_jobs=1, random_state=42, generations=20, population_size=300)
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp.fit(X_train, y_train)
                X_train_stgp = np.nan_to_num(stgp.transform(X_train))
                X_test_stgp = np.nan_to_num(stgp.transform(X_test))
                logger.info(f"STGP generated {X_train_stgp.shape[1]} symbolic features")
        except Exception as e:
            logger.error(f"STGP failed: {e}. Using original features for symbolic stage.")
            X_train_stgp = X_train.copy()
            X_test_stgp = X_test.copy()
        finally:
            del stgp
            gc.collect()

        # Stage 2: Evolutionary Forest
        logger.info("Stage 2: Applying Evolutionary Forest (EF)...")
        try:
            ef = EvolutionaryForestRegressor(
                random_state=42,
                basic_primitives="Default",
                n_gen=10,
                pop_size=50,
                verbose=False
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                ef.fit(X_train, y_train)
                X_train_ef = ef.transform(X_train)
                X_test_ef = ef.transform(X_test)

                # Select best features to avoid curse of dimensionality
                n_features_ef = X_train_ef.shape[1]
                n_select = min(n_best_features, n_features_ef)
                
                if n_select < n_features_ef:
                    logger.info(f"Selecting top {n_select} features from {n_features_ef} EF features")
                    X_train_ef = X_train_ef[:, :n_select]
                    X_test_ef = X_test_ef[:, :n_select]

                X_train_ef = np.nan_to_num(X_train_ef)
                X_test_ef = np.nan_to_num(X_test_ef)
                logger.info(f"EF generated {X_train_ef.shape[1]} evolutionary features")
        except Exception as e:
            logger.error(f"EF failed: {e}. Using original features for evolutionary stage.")
            X_train_ef = X_train.copy()
            X_test_ef = X_test.copy()

        # Stage 3: Hybrid Feature Matrix Construction
        logger.info("Stage 3: Constructing hybrid feature matrix...")
        X_train_hybrid = np.hstack((X_train, X_train_stgp, X_train_ef))
        X_test_hybrid = np.hstack((X_test, X_test_stgp, X_test_ef))
        
        logger.info(f"Original features: {X_train.shape[1]}, "
                   f"Symbolic features: {X_train_stgp.shape[1]}, "
                   f"Evolutionary features: {X_train_ef.shape[1]}, "
                   f"Total hybrid features: {X_train_hybrid.shape[1]}")

        # Stage 4: Algorithm Performance Evaluation
        logger.info("Stage 4: Evaluating algorithm performance...")
        dataset_results = evaluate_regressor_performance(
            X_train, y_train, X_test, y_test, 
            X_train_hybrid, X_test_hybrid, 
            dataset_name
        )
        all_results.extend(dataset_results)

    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        traceback.print_exc()
        return None, None

    if not all_results:
        logger.error("No results generated")
        return None, None

    # Build results dataframe
    logger.info("Processing results...")
    df_raw = pd.DataFrame(all_results)

    # Generate summary statistics
    stats_df = df_raw.groupby('Algorithm').agg({
        'Base_R2': 'mean',
        'Enhanced_R2': 'mean',
        'R2_Improvement': ['mean', 'std'],
        'Base_RMSE': 'mean',
        'Enhanced_RMSE': 'mean',
        'RMSE_Improvement': 'mean'
    }).round(4).reset_index()
    
    stats_df.columns = ['Algorithm', 'Mean_Base_R2', 'Mean_Enhanced_R2', 
                        'Mean_R2_Improvement', 'Std_R2_Improvement', 
                        'Mean_Base_RMSE', 'Mean_Enhanced_RMSE', 'Mean_RMSE_Improvement']

    logger.info("Feature engineering completed successfully!")
    return stats_df, df_raw