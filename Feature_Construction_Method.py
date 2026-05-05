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
from scipy.stats import spearmanr

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

def extract_symbolic_transformer_formulas(stgp_model, selected_indices: list = None) -> dict:
    """
    Extract and display formulas from Symbolic Transformer (STGP).
    
    Parameters:
    -----------
    stgp_model : SymbolicTransformer
        Fitted SymbolicTransformer model
    selected_indices : list
        Indices of selected features (optional - if None, shows all)
    
    Returns:
    --------
    dict : Dictionary mapping feature index to formula
    """
    formulas = {}
    try:
        if hasattr(stgp_model, '_programs'):
            programs = stgp_model._programs
            indices_to_show = selected_indices if selected_indices is not None else range(len(programs))
            
            for idx in indices_to_show:
                if idx < len(programs):
                    formula = str(programs[idx])
                    formulas[f'STGP_{idx}'] = formula
                    logger.info(f"  ✓ STGP_{idx}: {formula}")
        else:
            logger.warning("Could not extract STGP formulas - _programs attribute not found")
    except Exception as e:
        logger.warning(f"Error extracting STGP formulas: {e}")
    
    return formulas


def extract_ef_formulas(ef_model, selected_indices: list = None) -> dict:
    """
    Extract and display formulas from Evolutionary Forest.
    
    Parameters:
    -----------
    ef_model : EvolutionaryForestRegressor
        Fitted EvolutionaryForestRegressor model
    selected_indices : list
        Indices of selected features (optional - if None, shows first 10)
    
    Returns:
    --------
    dict : Dictionary mapping feature index to formula
    """
    formulas = {}
    try:
        if hasattr(ef_model, '_best_hof') or hasattr(ef_model, 'hof'):
            hof = getattr(ef_model, '_best_hof', getattr(ef_model, 'hof', None))
            if hof is not None:
                indices_to_show = selected_indices if selected_indices is not None else range(min(10, len(hof)))
                
                for idx in indices_to_show:
                    if idx < len(hof):
                        formula = str(hof[idx])
                        formulas[f'EF_{idx}'] = formula
                        logger.info(f"  ✓ EF_{idx}: {formula}")
        else:
            logger.warning("Could not extract EF formulas - hof attribute not found")
    except Exception as e:
        logger.warning(f"Error extracting EF formulas: {e}")
    
    return formulas


def remove_duplicate_features(X_train: np.ndarray, X_test: np.ndarray, 
                              feature_labels: list = None,
                              correlation_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Remove highly correlated and duplicate features from feature matrix.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    feature_labels : list
        Labels for features (for logging purposes)
    correlation_threshold : float
        Correlation threshold for considering features as duplicates (0-1)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, list]
        - Deduplicated X_train
        - Deduplicated X_test
        - Indices of retained features
        - Labels of retained features
    """
    n_features = X_train.shape[1]
    features_to_keep = []
    
    if feature_labels is None:
        feature_labels = [f'Feature_{i}' for i in range(n_features)]
    
    # Check each feature against already selected features
    for i in range(n_features):
        is_duplicate = False
        
        for j in features_to_keep:
            try:
                corr_coef, _ = spearmanr(X_train[:, i], X_train[:, j])
                
                if np.isnan(corr_coef):
                    corr_coef = 0
                
                if abs(corr_coef) >= correlation_threshold:
                    is_duplicate = True
                    logger.debug(f"  Duplicate: {feature_labels[i]} ↔ {feature_labels[j]} (r={corr_coef:.4f})")
                    break
            except Exception as e:
                logger.debug(f"Error comparing {feature_labels[i]} and {feature_labels[j]}: {e}")
                continue
        
        if not is_duplicate:
            features_to_keep.append(i)
    
    kept_labels = [feature_labels[i] for i in features_to_keep]
    logger.info(f"✓ Result: {n_features} → {len(features_to_keep)} unique features "
               f"(Removed {n_features - len(features_to_keep)})")
    # Filter features
    X_train_filtered = X_train[:, features_to_keep]
    X_test_filtered = X_test[:, features_to_keep]
    
    return X_train_filtered, X_test_filtered, np.array(features_to_keep), kept_labels

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

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1: Symbolic Regression via Genetic Programming
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 1/6: Fitting Symbolic Transformer (STGP)...")
        logger.info("="*80)
        
        stgp_formulas_all = {}
        X_train_stgp = None
        X_test_stgp = None
        stgp_model = None
        
        try:
            stgp_model = SymbolicTransformer(n_jobs=1, random_state=42, generations=20, population_size=300)
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp_model.fit(X_train, y_train)
                X_train_stgp = np.nan_to_num(stgp_model.transform(X_train))
                X_test_stgp = np.nan_to_num(stgp_model.transform(X_test))
                
                logger.info(f"✓ Generated {X_train_stgp.shape[1]} STGP features")
                logger.info("\nAll STGP formulas:")
                stgp_formulas_all = extract_symbolic_transformer_formulas(stgp_model)
        except Exception as e:
            logger.error(f"✗ STGP failed: {e}")
            X_train_stgp = X_train.copy()
            X_test_stgp = X_test.copy()
        finally:
            del stgp_model
            gc.collect()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2: Select Top-K STGP Features by Importance
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 2/6: Selecting Top-K STGP features by importance...")
        logger.info("="*80)
        
        top_stgp_indices = []
        try:
            rf_scorer = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
            rf_scorer.fit(X_train_stgp, y_train)
            importances = rf_scorer.feature_importances_
            
            n_stgp_selected = min(n_best_features, X_train_stgp.shape[1])
            top_stgp_indices = np.argsort(importances)[-n_stgp_selected:][::-1]
            
            X_train_stgp = X_train_stgp[:, top_stgp_indices]
            X_test_stgp = X_test_stgp[:, top_stgp_indices]
            
            logger.info(f"✓ Selected {n_stgp_selected} STGP features")
            logger.info("\nSelected STGP formulas:")
            for i, idx in enumerate(top_stgp_indices):
                formula = stgp_formulas_all.get(f'STGP_{idx}', 'N/A')
                logger.info(f"  {i+1}. STGP_{idx}: {formula} (importance: {importances[idx]:.4f})")
        except Exception as e:
            logger.error(f"✗ STGP selection failed: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3: Evolutionary Forest
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 3/6: Fitting Evolutionary Forest (EF)...")
        logger.info("="*80)
        
        ef_formulas_all = {}
        X_train_ef = None
        X_test_ef = None
        ef_model = None
        
        try:
            ef_model = EvolutionaryForestRegressor(
                random_state=42,
                basic_primitives="Default",
                n_gen=10,
                pop_size=50,
                verbose=False
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                ef_model.fit(X_train, y_train)
                X_train_ef = ef_model.transform(X_train)
                X_test_ef = ef_model.transform(X_test)
                
                logger.info(f"✓ Generated {X_train_ef.shape[1]} EF features")
                logger.info("\nTop EF formulas:")
                ef_formulas_all = extract_ef_formulas(ef_model)
        except Exception as e:
            logger.error(f"✗ EF failed: {e}")
            X_train_ef = X_train.copy()
            X_test_ef = X_test.copy()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 4: Select Top-K EF Features
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 4/6: Selecting Top-K EF features...")
        logger.info("="*80)
        
        try:
            n_ef_total = X_train_ef.shape[1]
            n_ef_selected = min(n_best_features, n_ef_total)
            
            X_train_ef = X_train_ef[:, :n_ef_selected]
            X_test_ef = X_test_ef[:, :n_ef_selected]
            
            logger.info(f"✓ Selected {n_ef_selected} EF features")
            logger.info("\nSelected EF formulas:")
            for i in range(n_ef_selected):
                formula = ef_formulas_all.get(f'EF_{i}', 'N/A')
                logger.info(f"  {i+1}. EF_{i}: {formula}")
        except Exception as e:
            logger.error(f"✗ EF selection failed: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 5: Combined Deduplication (STGP + EF)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 5/6: Combined Deduplication (STGP + EF)...")
        logger.info("="*80)
        
        try:
            X_train_combined = np.hstack((X_train_stgp, X_train_ef))
            X_test_combined = np.hstack((X_test_stgp, X_test_ef))
            
            stgp_labels = [f'STGP_{i}' for i in range(X_train_stgp.shape[1])]
            ef_labels = [f'EF_{i}' for i in range(X_train_ef.shape[1])]
            combined_labels = stgp_labels + ef_labels
            
            logger.info(f"Before dedup: {X_train_combined.shape[1]} features "
                       f"(STGP: {len(stgp_labels)}, EF: {len(ef_labels)})")
            X_train_constructed, X_test_constructed, _, kept_labels = remove_duplicate_features(
                X_train_combined, X_test_combined, 
                feature_labels=combined_labels,
                correlation_threshold=0.95
            )
            
            logger.info(f"Kept features: {', '.join(kept_labels)}")
            
        except Exception as e:
            logger.error(f"✗ Deduplication failed: {e}")
            X_train_constructed = X_train_combined.copy()
            X_test_constructed = X_test_combined.copy()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 6: Final Hybrid Matrix and Evaluation
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "="*80)
        logger.info("Stage 6/6: Algorithm Performance Evaluation...")
        logger.info("="*80)
        
        X_train_hybrid = np.hstack((X_train, X_train_constructed))
        X_test_hybrid = np.hstack((X_test, X_test_constructed))
        
        logger.info(f"\nFinal feature matrix:")
        logger.info(f"  Original: {X_train.shape[1]}")
        logger.info(f"  Constructed: {X_train_constructed.shape[1]}")
        logger.info(f"  Total: {X_train_hybrid.shape[1]}")
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