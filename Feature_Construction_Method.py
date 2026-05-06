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

import re
import gc
import logging
import traceback
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from evolutionary_forest.forest import EvolutionaryForestRegressor
from gplearn.genetic import SymbolicTransformer
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress numpy warnings
np.seterr(divide='ignore', invalid='ignore')

# ── GPLearn compatibility patch (sklearn 1.6+) ──────────────────────────────
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

# ── Evolutionary Forest compatibility patch ──────────────────────────────────
try:
    import evolutionary_forest.forest as _ef_mod
    _ef_mod.consistency_check = lambda learner: None
except (ImportError, AttributeError) as e:
    logger.warning(f"Evolutionary Forest compatibility patch failed: {e}")


def format_math_expr(expr: str) -> str:
    """
    Prefix (önek) formatındaki STGP ve EF formüllerini standart 
    matematiksel notasyona (infix) çevirir. 
    Örn: Add(ARG0, Mul(-1, ARG1)) -> (x0 + (-1 * x1))
    """
    # EF'deki ARG0, ARG1 gibi değişkenleri x0, x1 yap
    expr = re.sub(r'(?i)ARG(\d+)', r'x\1', expr)
    
    # STGP'deki X0, X1 gibi değişkenleri de x0, x1 yap
    expr = re.sub(r'\bX(\d+)\b', r'x\1', expr)
    
    # x0, x1 gibi değişkenleri eval() sırasında hata vermemesi için string içine al ("x0" gibi)
    expr = re.sub(r'\b(x\d+)\b', r'"\1"', expr)
    
    # Matematiksel operatörlerin standart karşılıklarını tanımla (Büyük/Küçük harf duyarlı)
    safe_dict = {
        'Add': lambda a, b: f"({a} + {b})",
        'add': lambda a, b: f"({a} + {b})",
        'Sub': lambda a, b: f"({a} - {b})",
        'sub': lambda a, b: f"({a} - {b})",
        'Mul': lambda a, b: f"({a} * {b})",
        'mul': lambda a, b: f"({a} * {b})",
        'Div': lambda a, b: f"({a} / {b})",
        'div': lambda a, b: f"({a} / {b})",
        'AQ':  lambda a, b: f"({a} / {b})",  # Analytic Quotient (korumalı bölme)
        'Sin': lambda a: f"sin({a})",
        'sin': lambda a: f"sin({a})",
        'Cos': lambda a: f"cos({a})",
        'cos': lambda a: f"cos({a})",
        'Exp': lambda a: f"exp({a})",
        'exp': lambda a: f"exp({a})",
        'Log': lambda a: f"log({a})",
        'log': lambda a: f"log({a})",
        'Abs': lambda a: f"abs({a})",
        'abs': lambda a: f"abs({a})",
        'Neg': lambda a: f"(-{a})",
        'neg': lambda a: f"(-{a})",
        'Inv': lambda a: f"(1 / {a})",
        'inv': lambda a: f"(1 / {a})",
        'Max': lambda a, b: f"max({a}, {b})",
        'max': lambda a, b: f"max({a}, {b})",
        'Min': lambda a, b: f"min({a}, {b})",
        'min': lambda a, b: f"min({a}, {b})"
    }
    
    try:
        # String'i eval() ile değerlendirip matematiksel ifadelere çeviriyoruz
        formatted_expr = eval(expr, {"__builtins__": {}}, safe_dict)
        return formatted_expr
    except Exception:
        # Eğer tanımlanmayan bir fonksiyon varsa orijinal haline geri dön (tırnakları temizleyerek)
        return expr.replace('"', '')


def extract_symbolic_transformer_formulas(stgp_model, n_features: int = 10) -> dict:
    """
    Extract and display formulas from Symbolic Transformer (STGP).
    """
    formulas = {}
    try:
        if hasattr(stgp_model, '_best_programs'):
            programs = stgp_model._best_programs
            n_to_show = min(n_features, len(programs))
            logger.info(f"\n{'─' * 78}")
            logger.info(f"SYMBOLIC REGRESSION (STGP) - Generated Features ({n_to_show} of {len(programs)})")
            logger.info(f"{'─' * 78}")
            for idx in range(n_to_show):
                raw_formula = str(programs[idx])
                # HATA DÜZELTMESİ: Matematiksel formata çevirici fonksiyon çağrıldı
                formula = format_math_expr(raw_formula)
                formulas[f'STGP_{idx}'] = formula
                logger.info(f"  STGP_{idx:02d}: {formula}")
        else:
            logger.warning("Could not extract STGP formulas — _best_programs attribute not found.")
    except Exception as e:
        logger.warning(f"Error extracting STGP formulas: {e}")
    return formulas


def extract_ef_formulas(ef_model, n_features: int = 10) -> dict:
    """
    Extract and display formulas from Evolutionary Forest.
    """
    formulas = {}
    try:
        if hasattr(ef_model, '_best_hof') or hasattr(ef_model, 'hof'):
            hof = getattr(ef_model, '_best_hof', getattr(ef_model, 'hof', None))
            if hof is not None:
                n_to_show = min(n_features, len(hof))
                logger.info(f"\n{'─' * 78}")
                logger.info(f"EVOLUTIONARY FOREST (EF) - Generated Features ({n_to_show} of {len(hof)})")
                logger.info(f"{'─' * 78}")
                for idx in range(n_to_show):
                    ind = hof[idx]
                    # HATA DÜZELTMESİ: Matematiksel formata çevirici fonksiyon çağrıldı
                    if isinstance(ind, (list, tuple)):
                        formula = " | ".join(format_math_expr(str(tree)) for tree in ind)
                    else:
                        formula = format_math_expr(str(ind))
                        
                    formulas[f'EF_{idx}'] = formula
                    logger.info(f"  EF_{idx:02d}: {formula}")
        else:
            logger.warning("Could not extract EF formulas — hof attribute not found.")
    except Exception as e:
        logger.warning(f"Error extracting EF formulas: {e}")
    return formulas


def remove_duplicate_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_labels: List[str] = None,
    correlation_threshold: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Remove highly correlated and duplicate features using a vectorised
    Spearman correlation matrix instead of a pairwise loop.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    feature_labels : list of str, optional
        Labels for features (used for logging).
    correlation_threshold : float
        Absolute correlation threshold above which a feature is considered
        a duplicate (0–1).

    Returns
    -------
    Tuple of (X_train_filtered, X_test_filtered, kept_indices, kept_labels)
    """
    n_features = X_train.shape[1]

    if feature_labels is None:
        feature_labels = [f'Feature_{i}' for i in range(n_features)]

    # Build full Spearman correlation matrix in one shot (vectorised)
    corr_matrix = (
        pd.DataFrame(X_train)
        .corr(method='spearman')
        .abs()
        .fillna(0)
        .values
    )

    # Upper-triangle mask — only compare each pair once
    upper = np.triu(np.ones((n_features, n_features), dtype=bool), k=1)

    features_to_keep = []
    dropped = set()

    for i in range(n_features):
        if i in dropped:
            continue
        features_to_keep.append(i)
        # Mark all features that are highly correlated with i as duplicates
        for j in range(i + 1, n_features):
            if upper[i, j] and corr_matrix[i, j] >= correlation_threshold:
                logger.debug(
                    f"  Duplicate: {feature_labels[j]} ↔ {feature_labels[i]} "
                    f"(r={corr_matrix[i, j]:.4f})"
                )
                dropped.add(j)

    kept_labels = [feature_labels[i] for i in features_to_keep]
    logger.info(
        f"✓ Result: {n_features} → {len(features_to_keep)} unique features "
        f"(Removed {n_features - len(features_to_keep)})"
    )

    X_train_filtered = X_train[:, features_to_keep]
    X_test_filtered = X_test[:, features_to_keep]

    return X_train_filtered, X_test_filtered, np.array(features_to_keep), kept_labels


def evaluate_regressor_performance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train_enhanced: np.ndarray,
    X_test_enhanced: np.ndarray,
    dataset_name: str,
) -> list:
    """
    Evaluate and compare regression model performance on original vs.
    enhanced (constructed) features.

    Parameters
    ----------
    X_train, y_train : Original training data.
    X_test, y_test : Original test data.
    X_train_enhanced, X_test_enhanced : Enhanced data with constructed features.
    dataset_name : str
        Name of the dataset (used for logging and result rows).

    Returns
    -------
    list of dict
        Performance metrics for each algorithm.
    """
    regressor_dict = {
        'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'ExtraTrees':   ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        'AdaBoost':     AdaBoostRegressor(n_estimators=100, random_state=42),
        'GBDT':         GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost':      XGBRegressor(n_jobs=1, n_estimators=100, verbosity=0, random_state=42),
        'LightGBM':     LGBMRegressor(n_jobs=1, n_estimators=100, verbose=-1, random_state=42),
        'CatBoost':     CatBoostRegressor(
                            n_estimators=100, thread_count=1,
                            verbose=False, allow_writing_files=False, random_state=42
                        ),
    }

    results = []

    for regr_name, regressor in regressor_dict.items():
        try:
            logger.info(f"Evaluating {regr_name} on {dataset_name}...")

            # Base model — original features
            base_model = clone(regressor)
            base_model.fit(X_train, y_train)
            y_pred_base = base_model.predict(X_test)

            base_r2   = r2_score(y_test, y_pred_base)
            base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
            base_mae  = mean_absolute_error(y_test, y_pred_base)

            # Enhanced model — constructed features
            enh_model = clone(regressor)
            enh_model.fit(X_train_enhanced, y_train)
            y_pred_enh = enh_model.predict(X_test_enhanced)

            enh_r2   = r2_score(y_test, y_pred_enh)
            enh_rmse = np.sqrt(mean_squared_error(y_test, y_pred_enh))
            enh_mae  = mean_absolute_error(y_test, y_pred_enh)

            results.append({
                'Algorithm':        regr_name,
                'Dataset':          dataset_name,
                'Base_R2':          base_r2,
                'Enhanced_R2':      enh_r2,
                'R2_Improvement':   enh_r2 - base_r2,
                'Base_RMSE':        base_rmse,
                'Enhanced_RMSE':    enh_rmse,
                'RMSE_Improvement': base_rmse - enh_rmse,
                'Base_MAE':         base_mae,
                'Enhanced_MAE':     enh_mae,
                'MAE_Improvement':  base_mae - enh_mae,
            })

        except Exception as e:
            logger.error(f"Error ({regr_name} - {dataset_name}): {e}")
            traceback.print_exc()
            results.append({
                'Algorithm': regr_name, 'Dataset': dataset_name,
                'Base_R2': 0, 'Enhanced_R2': 0, 'R2_Improvement': 0,
                'Base_RMSE': 0, 'Enhanced_RMSE': 0, 'RMSE_Improvement': 0,
                'Base_MAE': 0, 'Enhanced_MAE': 0, 'MAE_Improvement': 0,
            })

    return results


def sr_ef_feature_engineering(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "CustomDataset",
    n_best_features: int = 10,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Hybridized Feature Construction combining Symbolic Regression (STGP)
    and Evolutionary Forest (EF) for regression tasks.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target variable.
    dataset_name : str
        Dataset name used in result rows and logging.
    n_best_features : int
        Number of top features to select from each method.
    test_size : float
        Fraction of data held out for evaluation.

    Returns
    -------
    pd.DataFrame
        Detailed per-algorithm results (one row per algorithm per dataset).
        Returns an empty DataFrame if the pipeline fails.
    """
    logger.info(f"Starting feature engineering on {dataset_name}...")
    np.seterr(divide='ignore', invalid='ignore')

    all_results = []

    try:
        # ── Data preparation ─────────────────────────────────────────────────
        logger.info("Preparing data...")
        X_vals = (
            np.nan_to_num(X.values.astype(np.float64))
            if isinstance(X, pd.DataFrame)
            else np.nan_to_num(X.astype(np.float64))
        )
        y_vals = y.values if isinstance(y, pd.Series) else y
        y_vals = y_vals.astype(np.float64)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vals, y_vals, test_size=test_size, random_state=42
        )

        # ── Standardisation ──────────────────────────────────────────────────
        logger.info("Standardising features...")
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test  = scaler_x.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 1 — Symbolic Regression via Genetic Programming (STGP)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Stage 1/4: Fitting Symbolic Transformer (STGP)...")
        logger.info("=" * 80)

        stgp_formulas_all = {}
        # HATA DÜZELTMESİ: STGP başarısız olursa orijinal özelliklerin kopya olmasını engellemek için boş array atandı.
        X_train_stgp = np.empty((X_train.shape[0], 0))
        X_test_stgp  = np.empty((X_test.shape[0], 0))

        try:
            stgp_model = SymbolicTransformer(
                n_jobs=1, random_state=42
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                stgp_model.fit(X_train, y_train)
                X_train_stgp = np.nan_to_num(stgp_model.transform(X_train))
                X_test_stgp  = np.nan_to_num(stgp_model.transform(X_test))
                logger.info(f"✓ Generated {X_train_stgp.shape[1]} STGP features")
                stgp_formulas_all = extract_symbolic_transformer_formulas(stgp_model, n_features=n_best_features)
        except Exception as e:
            logger.error(f"✗ STGP failed: {e}")
        finally:
            # Release STGP model memory before proceeding
            try:
                del stgp_model
            except NameError:
                pass
            gc.collect()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 2 — Top STGP Features
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info(f"Stage 2/4: Top {n_best_features} STGP Features")
        logger.info("=" * 80)

        try:
            n_stgp_selected = min(n_best_features, X_train_stgp.shape[1])
            if n_stgp_selected > 0:
                top_stgp_indices = np.arange(n_stgp_selected)
                X_train_stgp = X_train_stgp[:, top_stgp_indices]
                X_test_stgp  = X_test_stgp[:, top_stgp_indices]
                logger.info(f"✓ Selected {n_stgp_selected} STGP features")
            else:
                logger.warning("No STGP features were generated/selected.")
        except Exception as e:
            logger.error(f"✗ STGP feature extraction failed: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 3 — Evolutionary Forest (EF)
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Stage 3/4: Fitting Evolutionary Forest (EF)...")
        logger.info("=" * 80)

        ef_formulas_all = {}
        # HATA DÜZELTMESİ: EF başarısız olursa orijinal özelliklerin kopya olmasını engellemek için boş array atandı.
        X_train_ef = np.empty((X_train.shape[0], 0))
        X_test_ef  = np.empty((X_test.shape[0], 0))

        try:
            ef_model = EvolutionaryForestRegressor(
                random_state=42,
                basic_primitives="Default",
                n_gen=10,
                pop_size=50,
                verbose=False,
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                ef_model.fit(X_train, y_train)
                X_train_ef = ef_model.transform(X_train)
                X_test_ef  = ef_model.transform(X_test)
                logger.info(f"✓ Generated {X_train_ef.shape[1]} EF features")
                ef_formulas_all = extract_ef_formulas(ef_model, n_features=n_best_features)
        except Exception as e:
            logger.error(f"✗ EF failed: {e}")
        finally:
            # Release EF model memory
            try:
                del ef_model
            except NameError:
                pass
            gc.collect()

        # ─────────────────────────────────────────────────────────────────────
        # Stage 4 — Top EF Features
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info(f"Stage 4/4: Top {n_best_features} EF Features")
        logger.info("=" * 80)

        try:
            n_ef_selected = min(n_best_features, X_train_ef.shape[1])
            if n_ef_selected > 0:
                X_train_ef = X_train_ef[:, :n_ef_selected]
                X_test_ef  = X_test_ef[:, :n_ef_selected]
                logger.info(f"✓ Selected {n_ef_selected} EF features")
            else:
                logger.warning("No EF features were generated/selected.")
        except Exception as e:
            logger.error(f"✗ EF selection failed: {e}")

        # ─────────────────────────────────────────────────────────────────────
        # Stage 5 — Combine Features + Evaluation
        # ─────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Stage 5/4: Combining features and evaluation...")
        logger.info("=" * 80)

        try:
            X_train_combined = np.hstack((X_train_stgp, X_train_ef))
            X_test_combined  = np.hstack((X_test_stgp, X_test_ef))

            stgp_labels = [f'STGP_{i}' for i in range(X_train_stgp.shape[1])]
            ef_labels   = [f'EF_{i}'   for i in range(X_train_ef.shape[1])]
            combined_labels = stgp_labels + ef_labels

            logger.info("\n✓ Combined Features:")
            logger.info(f"  STGP Features: {len(stgp_labels)}")
            logger.info(f"  EF Features:   {len(ef_labels)}")
            logger.info(f"  Total:         {X_train_combined.shape[1]}")
            if combined_labels:
                logger.info(f"\nFeature List: {', '.join(combined_labels)}")

            X_train_constructed = X_train_combined.copy()
            X_test_constructed  = X_test_combined.copy()

        except Exception as e:
            logger.error(f"✗ Feature combination failed: {e}")
            X_train_constructed = np.empty((X_train.shape[0], 0))
            X_test_constructed  = np.empty((X_test.shape[0], 0))

        # Orijinal matris ve yeni üretilenler (yatay olarak) birleştirilir
        X_train_hybrid = np.hstack((X_train, X_train_constructed))
        X_test_hybrid  = np.hstack((X_test, X_test_constructed))

        logger.info("\nFinal feature matrix:")
        logger.info(f"  Original:    {X_train.shape[1]}")
        logger.info(f"  Constructed: {X_train_constructed.shape[1]}")
        logger.info(f"  Total:       {X_train_hybrid.shape[1]}")

        # Daha önce tanımlanan performans fonksiyonu çağrılıyor
        dataset_results = evaluate_regressor_performance(
            X_train, y_train, X_test, y_test,
            X_train_hybrid, X_test_hybrid,
            dataset_name,
        )
        all_results.extend(dataset_results)

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    if not all_results:
        logger.error("No results generated.")
        return pd.DataFrame()

    logger.info("Feature engineering completed successfully!")
    return pd.DataFrame(all_results)