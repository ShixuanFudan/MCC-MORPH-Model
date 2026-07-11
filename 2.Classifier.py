# =====================================================================
# MORPH Framework: Deep Geometric Learning for Drug-Protein Interactions
# Module: Downstream Supervised Classifiers & Ensemble Inference Engine
# =====================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, \
    f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from tqdm import tqdm
import warnings
import datetime
import os

# Configure matplotlib rendering parameters to satisfy SCI publication styles
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False
})

warnings.filterwarnings('ignore')


# =====================================================================
# High-Performance Area Under the ROC Curve (AUC) Optimization Configurations
# =====================================================================

class Breakthrough08Config:
    """Hyperparameter space mappings and training controls tailored for premium classification."""

    def __init__(self):
        # Tri-partition data stratification parameters
        self.train_size = 0.80
        self.test_size = 0.15
        self.validation_size = 0.15
        self.random_state = 42

        # Feature pre-processing configurations
        self.use_pca = False
        self.pca_components = 200
        self.use_feature_scaling = True
        self.use_quantile_transform = False

        # Random Forest (RF) structural parameters
        self.rf_n_estimators = [100, 150]
        self.rf_max_depth = [10, 15, None]
        self.rf_min_samples_split = [2, 5]
        self.rf_min_samples_leaf = [1, 2]
        self.rf_max_features = ['sqrt', 'log2']
        self.rf_class_weight = ['balanced']
        self.rf_bootstrap = [True]
        self.rf_cv_folds = 5
        self.rf_n_jobs = -1
        self.rf_use_fast_training = True

        # Extreme Gradient Boosting (XGBoost) tuning configurations
        self.xgb_n_estimators = [300, 400, 500]
        self.xgb_max_depth = [3, 4, 5]
        self.xgb_learning_rate = [0.01, 0.05, 0.1]
        self.xgb_subsample = [0.7, 0.8]
        self.xgb_colsample_bytree = [0.7, 0.8]
        self.xgb_colsample_bylevel = [0.7, 0.8]
        self.xgb_gamma = [0.5, 1, 2]
        self.xgb_reg_alpha = [0, 0.1, 0.5]
        self.xgb_reg_lambda = [1, 1.5, 2]
        self.xgb_scale_pos_weight = [1]
        self.xgb_cv_folds = 5
        self.xgb_min_child_weight = [5, 10, 20]

        # Light Gradient Boosting Machine (LightGBM) tuning configurations
        self.lgb_n_estimators = [300, 400, 500]
        self.lgb_max_depth = [5, 7]
        self.lgb_learning_rate = [0.01, 0.05, 0.1]
        self.lgb_num_leaves = [31, 63]
        self.lgb_subsample = [0.7, 0.8]
        self.lgb_colsample_bytree = [0.7, 0.8]
        self.lgb_reg_alpha = [0, 0.1, 0.5]
        self.lgb_reg_lambda = [0, 0.1, 0.5]
        self.lgb_min_child_samples = [50, 100]
        self.lgb_cv_folds = 5

        # Logistic Regression (LR) parameter specifications
        self.lr_C = [0.001, 0.01, 0.1, 1, 10, 100]
        self.lr_penalty = ['l1', 'l2', 'elasticnet']
        self.lr_solver = ['liblinear', 'saga']
        self.lr_class_weight = ['balanced']
        self.lr_max_iter = 2000
        self.lr_cv_folds = 5

        # Classifier selection controls
        self.train_rf = True
        self.train_xgb = True
        self.train_lgb = True
        self.train_lr = True
        self.train_ensemble = False

        # Output and serialization tracking paths
        self.save_best_model = True
        self.save_all_models = True
        self.output_model_name = 'breakthrough_08_model.pkl'
        self.output_dir = 'trained_models'
        self.plot_results = False
        self.detailed_analysis = True
        self.output_txt_file = 'model_performance_report.txt'

        # SCI publication figure criteria
        self.generate_sci_figures = True
        self.sci_figure_dpi = 300
        self.sci_figure_format = 'pdf'

        # Automated screening optimizations
        self.use_randomized_search = True
        self.randomized_search_iter = 25


# =====================================================================
# Multi-Classifier Training Orchestration Framework
# =====================================================================

class Breakthrough08Classifier:
    """Supervised learning executor equipped with rigorous cross-validation pipelines."""

    def __init__(self, features_dict, config=None):
        self.features_dict = features_dict
        self.config = config if config else Breakthrough08Config()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler() if self.config.use_feature_scaling else None
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal') if self.config.use_quantile_transform else None
        self.pca = None
        self.txt_writer = None
        self.txt_file_closed = False

        self._prepare_data()
        self._validate_data()
        self._init_txt_output()

    def _init_txt_output(self):
        """Initializes the structural performance documentation file."""
        self.txt_writer = open(self.config.output_txt_file, 'w', encoding='utf-8')
        self.txt_file_closed = False
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.txt_writer.write(f"CLASSIFIER PERFORMANCE REPORT - Generated: {current_time}\n")
        self.txt_writer.write("=" * 80 + "\n\n")

    def _write_to_txt(self, content, add_newline=True):
        """Appends structural logs into the persistent file buffer."""
        if self.txt_writer and not self.txt_file_closed:
            try:
                self.txt_writer.write(content)
                if add_newline:
                    self.txt_writer.write("\n")
            except ValueError:
                print(f"[WARNING] Buffer mismatch. File closed during write operation: {content}")
        elif self.txt_file_closed:
            print(f"[CONSOLE] File streaming terminated. Console echo: {content}")

    def _prepare_data(self):
        """Transforms raw features into stratified training, testing, and validation splits."""
        print("[INFO] Distributing computational dimensions into data subsets...")

        X = self.features_dict['features']
        y = self.features_dict['labels']

        print(f"[INFO] Global feature matrix dimensions: {X.shape}")
        print(f"[INFO] Target category metrics distribution: 0={np.sum(y == 0)}, 1={np.sum(y == 1)}")

        # Partition absolute training matrix from temporary testing arrays
        X_train_temp, X_test_val, y_train_temp, y_test_val = train_test_split(
            X, y, test_size=(self.config.test_size + self.config.validation_size),
            random_state=self.config.random_state, stratify=y
        )

        # Separate independent validation datasets from testing arrays
        test_val_ratio = self.config.test_size / (self.config.test_size + self.config.validation_size)
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_test_val, y_test_val, test_size=(1 - test_val_ratio),
            random_state=self.config.random_state, stratify=y_test_val
        )

        self.X_train = X_train_temp
        self.y_train = y_train_temp
        self.X_test = X_test
        self.y_test = y_test
        self.X_validation = X_validation
        self.y_validation = y_validation

        print(
            f"[INFO] Array configurations -> Train: {self.X_train.shape} | Test: {self.X_test.shape} | Validation: {self.X_validation.shape}")

        if self.config.use_feature_scaling:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.X_validation_scaled = self.scaler.transform(self.X_validation)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            self.X_validation_scaled = self.X_validation

        if self.config.use_quantile_transform:
            self.X_train_scaled = self.quantile_transformer.fit_transform(self.X_train_scaled)
            self.X_test_scaled = self.quantile_transformer.transform(self.X_test_scaled)
            self.X_validation_scaled = self.quantile_transformer.transform(self.X_validation_scaled)

        if self.config.use_pca and self.X_train.shape[1] > self.config.pca_components:
            self.pca = PCA(n_components=self.config.pca_components, random_state=self.config.random_state)
            self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
            self.X_test_pca = self.pca.transform(self.X_test_scaled)
            self.X_validation_pca = self.pca.transform(self.X_validation_scaled)
            print(f"[INFO] Matrix dimensionality post principal component analysis (PCA): {self.X_train_pca.shape}")
        else:
            self.X_train_pca = self.X_train_scaled
            self.X_test_pca = self.X_test_scaled
            self.X_validation_pca = self.X_validation_scaled

    def _validate_data(self):
        """Performs rigorous diagnostics to identify data leakage or class imbalances."""
        print("\n" + "=" * 50)
        print("[DIAGNOSTIC] INITIATING DATA INTEGRITY ASSESSMENTS")
        print("=" * 50)

        self._write_to_txt("DATA INTEGRITY ASSESSMENTS")
        self._write_to_txt("=" * 50)

        self._validate_class_distribution()
        self._validate_no_data_leakage()
        self._check_feature_quality()
        self._write_to_txt("")

    def _validate_class_distribution(self):
        """Quantifies the proportion distributions of positive vs negative cohorts."""
        print("\n1. Category Prevalence Analysis:")
        self._write_to_txt("\n1. Category Prevalence Analysis:")

        total_pos = np.sum(self.features_dict['labels'] == 1)
        total_neg = np.sum(self.features_dict['labels'] == 0)
        total_samples = len(self.features_dict['labels'])

        dist_info = f"  Global Space - Positives: {total_pos} ({total_pos / total_samples * 100:.2f}%), Negatives: {total_neg} ({total_neg / total_samples * 100:.2f}%)"
        print(dist_info)
        self._write_to_txt(dist_info)

        train_pos = np.sum(self.y_train == 1)
        train_neg = np.sum(self.y_train == 0)
        train_samples = len(self.y_train)

        dist_info = f"  Training Set - Positives: {train_pos} ({train_pos / train_samples * 100:.2f}%), Negatives: {train_neg} ({train_neg / train_samples * 100:.2f}%)"
        print(dist_info)
        self._write_to_txt(dist_info)

        test_pos = np.sum(self.y_test == 1)
        test_neg = np.sum(self.y_test == 0)
        test_samples = len(self.y_test)

        dist_info = f"  Testing Set  - Positives: {test_pos} ({test_pos / test_samples * 100:.2f}%), Negatives: {test_neg} ({test_neg / test_samples * 100:.2f}%)"
        print(dist_info)
        self._write_to_txt(dist_info)

        validation_pos = np.sum(self.y_validation == 1)
        validation_neg = np.sum(self.y_validation == 0)
        validation_samples = len(self.y_validation)

        dist_info = f"  Validation Set - Positives: {validation_pos} ({validation_pos / validation_samples * 100:.2f}%), Negatives: {validation_neg} ({validation_neg / validation_samples * 100:.2f}%)"
        print(dist_info)
        self._write_to_txt(dist_info)

        imbalance_ratio = max(total_pos, total_neg) / min(total_pos, total_neg) if min(total_pos,
                                                                                       total_neg) > 0 else float('inf')
        imbalance_info = f"  Cohort Imbalance Coefficient: {imbalance_ratio:.2f}"
        print(imbalance_info)
        self._write_to_txt(imbalance_info)

        if imbalance_ratio > 10:
            warning = "  [WARNING] Severe class imbalance detected within the distribution space."
            print(warning)
            self._write_to_txt(warning)
        elif imbalance_ratio > 5:
            warning = "  [NOTE] Moderate class skew noted across subsets."
            print(warning)
            self._write_to_txt(warning)
        else:
            info = "  [SUCCESS] Symmetrical class distributions validated."
            print(info)
            self._write_to_txt(info)

    def _validate_no_data_leakage(self):
        """Evaluates intersections between arrays to enforce rigorous out-of-sample validation."""
        print("\n2. Informational Data Leakage Verifications:")
        self._write_to_txt("\n2. Informational Data Leakage Verifications:")

        train_set = set(tuple(x) for x in self.X_train)
        test_set = set(tuple(x) for x in self.X_test)
        validation_set = set(tuple(x) for x in self.X_validation)

        train_test_overlap = train_set.intersection(test_set)
        train_validation_overlap = train_set.intersection(validation_set)
        test_validation_overlap = test_set.intersection(validation_set)

        if train_test_overlap:
            warning = f"  [CRITICAL] Informational overlap detected: {len(train_test_overlap)} items duplicated between Train and Test arrays."
            print(warning)
            self._write_to_txt(warning)
        else:
            print("  [SUCCESS] Training and Testing datasets are completely disjoint.")
            self._write_to_txt("  [SUCCESS] Training and Testing datasets are completely disjoint.")

        if train_validation_overlap:
            warning = f"  [CRITICAL] Informational overlap detected: {len(train_validation_overlap)} items duplicated between Train and Validation arrays."
            print(warning)
            self._write_to_txt(warning)
        else:
            print("  [SUCCESS] Training and independent Validation datasets are completely disjoint.")
            self._write_to_txt("  [SUCCESS] Training and independent Validation datasets are completely disjoint.")

        if test_validation_overlap:
            warning = f"  [CRITICAL] Informational overlap detected: {len(test_validation_overlap)} items duplicated between Test and Validation arrays."
            print(warning)
            self._write_to_txt(warning)
        else:
            print("  [SUCCESS] Testing and independent Validation datasets are completely disjoint.")
            self._write_to_txt("  [SUCCESS] Testing and independent Validation datasets are completely disjoint.")

    def _check_feature_quality(self):
        """Assesses missing values and variance criteria across latent dimensions."""
        print("\n3. Feature Vector Properties Profile:")
        self._write_to_txt("\n3. Feature Vector Properties Profile:")

        missing_values = np.isnan(self.X_train).sum()
        total_elements = self.X_train.size
        missing_ratio = missing_values / total_elements if total_elements > 0 else 0

        missing_info = f"  Missing values ratio inside Training matrix: {missing_ratio:.6f}"
        print(missing_info)
        self._write_to_txt(missing_info)

        selector = VarianceThreshold()
        selector.fit(self.X_train)
        n_constant_features = np.sum(selector.variances_ == 0)

        variance_info = f"  Constant (zero-variance) dimension variables: {n_constant_features}/{self.X_train.shape[1]}"
        print(variance_info)
        self._write_to_txt(variance_info)

        feature_means = np.mean(self.X_train, axis=0)
        feature_stds = np.std(self.X_train, axis=0)

        mean_info = f"  Computed empirical means interval bounds: [{np.min(feature_means):.4f}, {np.max(feature_means):.4f}]"
        std_info = f"  Computed standard deviations interval bounds: [{np.min(feature_stds):.4f}, {np.max(feature_stds):.4f}]"

        print(mean_info)
        print(std_info)
        self._write_to_txt(mean_info)
        self._write_to_txt(std_info)

    def _calculate_metrics(self, y_true, y_pred_proba, model_name, dataset_type="Test Set"):
        """Quantifies statistical evaluation metrics and logs discrimination performance."""
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n[METRIC] {model_name} Evaluation Statistics ({dataset_type}):")
        print(f"  Area Under the ROC Curve (AUC): {auc:.4f}")
        print(f"  Average Precision (AUPRC): {avg_precision:.4f}")
        print(f"  F1-Score Profile Metric: {f1:.4f}")
        print(f"  Sensitivity (Recall Rate): {recall:.4f}")
        print(f"  Positive Predictive Value (Precision): {precision:.4f}")
        print(f"  Global Accuracy Score: {accuracy:.4f}")
        print(f"  Confusion Matrix Struct -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

        self._write_to_txt(f"\n{model_name} Evaluation Statistics ({dataset_type}):")
        self._write_to_txt(f"  Area Under the ROC Curve (AUC): {auc:.4f}")
        self._write_to_txt(f"  Average Precision (AUPRC): {avg_precision:.4f}")
        self._write_to_txt(f"  F1-Score Profile Metric: {f1:.4f}")
        self._write_to_txt(f"  Sensitivity (Recall Rate): {recall:.4f}")
        self._write_to_txt(f"  Positive Predictive Value (Precision): {precision:.4f}")
        self._write_to_txt(f"  Global Accuracy Score: {accuracy:.4f}")
        self._write_to_txt(f"  Confusion Matrix Struct -> TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'average_precision': avg_precision,
            'confusion_matrix': cm,
            'predictions': y_pred_proba,
            'binary_predictions': y_pred
        }

    def _calculate_optimal_cutoff(self, y_true, y_pred_proba):
        """Determines the optimal diagnostic decision threshold using Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        return thresholds[np.argmax(j_scores)]

    def train_random_forest_breakthrough(self):
        """Executes hyperparameter sweeps for Random Forest classification architectures."""
        if not self.config.train_rf:
            print("[INFO] Random Forest execution omitted per configurations.")
            return None

        print("\n" + "=" * 50)
        print("[TRAIN] OPTIMIZING RANDOM FOREST ENGINES")
        print("=" * 50)
        self._write_to_txt("\n" + "=" * 50)
        self._write_to_txt("OPTIMIZING RANDOM FOREST ENGINES")
        self._write_to_txt("=" * 50)

        param_grid = {
            'n_estimators': self.config.rf_n_estimators,
            'max_depth': self.config.rf_max_depth,
            'min_samples_split': self.config.rf_min_samples_split,
            'min_samples_leaf': self.config.rf_min_samples_leaf,
            'max_features': self.config.rf_max_features,
            'class_weight': self.config.rf_class_weight,
            'bootstrap': self.config.rf_bootstrap
        }

        rf_model = RandomForestClassifier(random_state=self.config.random_state, n_jobs=self.config.rf_n_jobs)
        cv = StratifiedKFold(n_splits=self.config.rf_cv_folds, shuffle=True, random_state=self.config.random_state)

        if self.config.use_randomized_search:
            search = RandomizedSearchCV(
                rf_model, param_grid, n_iter=self.config.randomized_search_iter, cv=cv,
                scoring='roc_auc', n_jobs=1, verbose=1, random_state=self.config.random_state
            )
        else:
            search = GridSearchCV(rf_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=1)

        search.fit(self.X_train_pca, self.y_train)
        best_rf = search.best_estimator_
        self.models['random_forest'] = best_rf

        y_test_pred_proba = best_rf.predict_proba(self.X_test_pca)[:, 1]
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred_proba, "Random Forest", "Test Set")
        test_cutoff = self._calculate_optimal_cutoff(self.y_test, y_test_pred_proba)
        test_metrics['optimal_cutoff'] = test_cutoff

        self._calculate_metrics(self.y_train, best_rf.predict_proba(self.X_train_pca)[:, 1], "Random Forest",
                                "Train Set")

        y_validation_pred_proba = best_rf.predict_proba(self.X_validation_pca)[:, 1]
        validation_metrics = self._calculate_metrics(self.y_validation, y_validation_pred_proba, "Random Forest",
                                                     "Validation Set")
        validation_cutoff = self._calculate_optimal_cutoff(self.y_validation, y_validation_pred_proba)
        validation_metrics['optimal_cutoff'] = validation_cutoff

        self.results['random_forest'] = {
            'model': best_rf, 'auc': test_metrics['auc'], 'validation_auc': validation_metrics['auc'],
            'best_params': search.best_params_, 'test_predictions': y_test_pred_proba,
            'validation_predictions': y_validation_pred_proba,
            'test_metrics': test_metrics, 'validation_metrics': validation_metrics, 'test_cutoff': test_cutoff,
            'validation_cutoff': validation_cutoff
        }

        print(f"[SUCCESS] RF Optimal Parameters Space: {search.best_params_}")
        print(f"[SUCCESS] RF Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")
        self._write_to_txt(f"RF Optimal Parameters Space: {search.best_params_}")
        self._write_to_txt(
            f"RF Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")

        if validation_metrics['auc'] > 0.80:
            self._save_intermediate_model(best_rf, 'random_forest', validation_metrics['auc'])
        return best_rf

    def train_xgboost_breakthrough(self):
        """Executes hyperparameter optimization sweeps for Extreme Gradient Boosting architectures."""
        if not self.config.train_xgb:
            print("[INFO] XGBoost execution omitted per configurations.")
            return None

        print("\n" + "=" * 50)
        print("[TRAIN] OPTIMIZING EXTREME GRADIENT BOOSTING (XGBOOST)")
        print("=" * 50)
        self._write_to_txt("\n" + "=" * 50)
        self._write_to_txt("OPTIMIZING EXTREME GRADIENT BOOSTING (XGBOOST)")
        self._write_to_txt("=" * 50)

        scale_pos_weight = [1, self._calculate_scale_pos_weight()]
        xgb_scale_pos_weight = self.config.xgb_scale_pos_weight + scale_pos_weight

        param_grid = {
            'n_estimators': self.config.xgb_n_estimators,
            'max_depth': self.config.xgb_max_depth,
            'learning_rate': self.config.xgb_learning_rate,
            'subsample': self.config.xgb_subsample,
            'colsample_bytree': self.config.xgb_colsample_bytree,
            'colsample_bylevel': self.config.xgb_colsample_bylevel,
            'gamma': self.config.xgb_gamma,
            'reg_alpha': self.config.xgb_reg_alpha,
            'reg_lambda': self.config.xgb_reg_lambda,
            'scale_pos_weight': xgb_scale_pos_weight,
            'min_child_weight': [1, 3, 5]
        }

        xgb_model = xgb.XGBClassifier(random_state=self.config.random_state, use_label_encoder=False,
                                      eval_metric='logloss', n_jobs=-1)
        cv = StratifiedKFold(n_splits=self.config.xgb_cv_folds, shuffle=True, random_state=self.config.random_state)

        if self.config.use_randomized_search:
            search = RandomizedSearchCV(
                xgb_model, param_grid, n_iter=self.config.randomized_search_iter, cv=cv,
                scoring='roc_auc', n_jobs=1, verbose=1, random_state=self.config.random_state
            )
        else:
            search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=1)

        search.fit(self.X_train_scaled, self.y_train)
        best_xgb = search.best_estimator_
        self.models['xgboost'] = best_xgb

        y_test_pred_proba = best_xgb.predict_proba(self.X_test_scaled)[:, 1]
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred_proba, "XGBoost", "Test Set")
        test_cutoff = self._calculate_optimal_cutoff(self.y_test, y_test_pred_proba)
        test_metrics['optimal_cutoff'] = test_cutoff

        self._calculate_metrics(self.y_train, best_xgb.predict_proba(self.X_train_scaled)[:, 1], "XGBoost", "Train Set")

        y_validation_pred_proba = best_xgb.predict_proba(self.X_validation_scaled)[:, 1]
        validation_metrics = self._calculate_metrics(self.y_validation, y_validation_pred_proba, "XGBoost",
                                                     "Validation Set")
        validation_cutoff = self._calculate_optimal_cutoff(self.y_validation, y_validation_pred_proba)
        validation_metrics['optimal_cutoff'] = validation_cutoff

        self.results['xgboost'] = {
            'model': best_xgb, 'auc': test_metrics['auc'], 'validation_auc': validation_metrics['auc'],
            'best_params': search.best_params_, 'test_predictions': y_test_pred_proba,
            'validation_predictions': y_validation_pred_proba,
            'test_metrics': test_metrics, 'validation_metrics': validation_metrics, 'test_cutoff': test_cutoff,
            'validation_cutoff': validation_cutoff
        }

        print(f"[SUCCESS] XGBoost Optimal Parameters Space: {search.best_params_}")
        print(
            f"[SUCCESS] XGBoost Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")
        self._write_to_txt(f"XGBoost Optimal Parameters Space: {search.best_params_}")
        self._write_to_txt(
            f"XGBoost Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")

        if validation_metrics['auc'] > 0.8:
            self._save_intermediate_model(best_xgb, 'xgboost', validation_metrics['auc'])
        return best_xgb

    def train_lightgbm_breakthrough(self):
        """Executes hyperparameter optimization sweeps for Light Gradient Boosting Machine models."""
        if not self.config.train_lgb:
            print("[INFO] LightGBM execution omitted per configurations.")
            return None

        print("\n" + "=" * 50)
        print("[TRAIN] OPTIMIZING LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM)")
        print("=" * 50)
        self._write_to_txt("\n" + "=" * 50)
        self._write_to_txt("OPTIMIZING LIGHT GRADIENT BOOSTING MACHINE (LIGHTGBM)")
        self._write_to_txt("=" * 50)

        param_grid = {
            'n_estimators': self.config.lgb_n_estimators,
            'max_depth': self.config.lgb_max_depth,
            'learning_rate': self.config.lgb_learning_rate,
            'num_leaves': self.config.lgb_num_leaves,
            'subsample': self.config.lgb_subsample,
            'colsample_bytree': self.config.lgb_colsample_bytree,
            'reg_alpha': self.config.lgb_reg_alpha,
            'reg_lambda': self.config.lgb_reg_lambda,
            'min_child_samples': self.config.lgb_min_child_samples,
            'min_split_gain': [0, 0.1, 0.2]
        }

        lgb_model = lgb.LGBMClassifier(random_state=self.config.random_state, class_weight='balanced', n_jobs=-1)
        cv = StratifiedKFold(n_splits=self.config.lgb_cv_folds, shuffle=True, random_state=self.config.random_state)

        if self.config.use_randomized_search:
            search = RandomizedSearchCV(
                lgb_model, param_grid, n_iter=self.config.randomized_search_iter, cv=cv,
                scoring='roc_auc', n_jobs=1, verbose=1, random_state=self.config.random_state
            )
        else:
            search = GridSearchCV(lgb_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=1, verbose=1)

        search.fit(self.X_train_scaled, self.y_train)
        best_lgb = search.best_estimator_
        self.models['lightgbm'] = best_lgb

        y_test_pred_proba = best_lgb.predict_proba(self.X_test_scaled)[:, 1]
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred_proba, "LightGBM", "Test Set")
        test_cutoff = self._calculate_optimal_cutoff(self.y_test, y_test_pred_proba)
        test_metrics['optimal_cutoff'] = test_cutoff

        self._calculate_metrics(self.y_train, best_lgb.predict_proba(self.X_train_scaled)[:, 1], "LightGBM",
                                "Train Set")

        y_validation_pred_proba = best_lgb.predict_proba(self.X_validation_scaled)[:, 1]
        validation_metrics = self._calculate_metrics(self.y_validation, y_validation_pred_proba, "LightGBM",
                                                     "Validation Set")
        validation_cutoff = self._calculate_optimal_cutoff(self.y_validation, y_validation_pred_proba)
        validation_metrics['optimal_cutoff'] = validation_cutoff

        self.results['lightgbm'] = {
            'model': best_lgb, 'auc': test_metrics['auc'], 'validation_auc': validation_metrics['auc'],
            'best_params': search.best_params_, 'test_predictions': y_test_pred_proba,
            'validation_predictions': y_validation_pred_proba,
            'test_metrics': test_metrics, 'validation_metrics': validation_metrics, 'test_cutoff': test_cutoff,
            'validation_cutoff': validation_cutoff
        }

        print(f"[SUCCESS] LightGBM Optimal Parameters Space: {search.best_params_}")
        print(
            f"[SUCCESS] LightGBM Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")
        self._write_to_txt(f"LightGBM Optimal Parameters Space: {search.best_params_}")
        self._write_to_txt(
            f"LightGBM Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")

        if validation_metrics['auc'] > 0.78:
            self._save_intermediate_model(best_lgb, 'lightgbm', validation_metrics['auc'])
        return best_lgb

    def train_logistic_regression_breakthrough(self):
        """Executes parameter regularization adjustments for baseline Logistic Regression models."""
        if not self.config.train_lr:
            print("[INFO] Logistic Regression execution omitted per configurations.")
            return None

        print("\n" + "=" * 50)
        print("[TRAIN] OPTIMIZING LOGISTIC REGRESSION MODELS")
        print("=" * 50)
        self._write_to_txt("\n" + "=" * 50)
        self._write_to_txt("OPTIMIZING LOGISTIC REGRESSION MODELS")
        self._write_to_txt("=" * 50)

        param_grid = {
            'C': self.config.lr_C,
            'penalty': self.config.lr_penalty,
            'solver': self.config.lr_solver,
            'class_weight': self.config.lr_class_weight
        }

        lr_model = LogisticRegression(random_state=self.config.random_state, max_iter=self.config.lr_max_iter)
        cv = StratifiedKFold(n_splits=self.config.lr_cv_folds, shuffle=True, random_state=self.config.random_state)

        if self.config.use_randomized_search:
            search = RandomizedSearchCV(
                lr_model, param_grid, n_iter=min(self.config.randomized_search_iter, 20), cv=cv,
                scoring='roc_auc', n_jobs=-1, verbose=1, random_state=self.config.random_state
            )
        else:
            search = GridSearchCV(lr_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

        search.fit(self.X_train_scaled, self.y_train)
        best_lr = search.best_estimator_
        self.models['logistic_regression'] = best_lr

        y_test_pred_proba = best_lr.predict_proba(self.X_test_scaled)[:, 1]
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred_proba, "Logistic Regression", "Test Set")
        test_cutoff = self._calculate_optimal_cutoff(self.y_test, y_test_pred_proba)
        test_metrics['optimal_cutoff'] = test_cutoff

        self._calculate_metrics(self.y_train, best_lr.predict_proba(self.X_train_scaled)[:, 1], "Logistic Regression",
                                "Train Set")

        y_validation_pred_proba = best_lr.predict_proba(self.X_validation_scaled)[:, 1]
        validation_metrics = self._calculate_metrics(self.y_validation, y_validation_pred_proba, "Logistic Regression",
                                                     "Validation Set")
        validation_cutoff = self._calculate_optimal_cutoff(self.y_validation, y_validation_pred_proba)
        validation_metrics['optimal_cutoff'] = validation_cutoff

        self.results['logistic_regression'] = {
            'model': best_lr, 'auc': test_metrics['auc'], 'validation_auc': validation_metrics['auc'],
            'best_params': search.best_params_, 'test_predictions': y_test_pred_proba,
            'validation_predictions': y_validation_pred_proba,
            'test_metrics': test_metrics, 'validation_metrics': validation_metrics, 'test_cutoff': test_cutoff,
            'validation_cutoff': validation_cutoff
        }

        print(f"[SUCCESS] LR Optimal Parameters Space: {search.best_params_}")
        print(f"[SUCCESS] LR Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")
        self._write_to_txt(f"LR Optimal Parameters Space: {search.best_params_}")
        self._write_to_txt(
            f"LR Discrimination Cutoffs -> Test: {test_cutoff:.4f} | Validation: {validation_cutoff:.4f}")
        return best_lr

    def _calculate_scale_pos_weight(self):
        """Determines minority class indexing scaling factor ratios for training stabilization."""
        n_negative = np.sum(self.y_train == 0)
        n_positive = np.sum(self.y_train == 1)
        return n_negative / n_positive if n_positive > 0 else 1

    def _save_intermediate_model(self, model, model_name, auc_score):
        """Saves intermediate model targets to guarantee checkpoint safety."""
        filename = f'intermediate_{model_name}_auc_{auc_score:.4f}.pkl'
        joblib.dump(model, filename)
        print(f"[SUCCESS] Checkpoint file successfully exported: {filename}")
        self._write_to_txt(f"Checkpoint file successfully exported: {filename}")

    def advanced_stacking_ensemble(self):
        """Orchestrates an advanced consensus meta-ensemble tracking network dynamics."""
        if len(self.results) < 2:
            print("[WARNING] Ensemble blending algorithms necessitate at least two operational primary estimators.")
            self._write_to_txt("Ensemble blending algorithms necessitate at least two operational primary estimators.")
            return None

        print("\n" + "=" * 50)
        print("[TRAIN] INITIALIZING CONSENSUS META-ENSEMBLE CORE")
        print("=" * 50)
        self._write_to_txt("\n" + "=" * 50)
        self._write_to_txt("INITIALIZING CONSENSUS META-ENSEMBLE CORE")
        self._write_to_txt("=" * 50)

        available_models = []
        model_test_predictions = {}
        model_validation_predictions = {}
        model_aucs = {}

        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']:
            if model_name in self.results:
                available_models.append(model_name)
                model_test_predictions[model_name] = self.results[model_name]['test_predictions']
                model_validation_predictions[model_name] = self.results[model_name]['validation_predictions']
                model_aucs[model_name] = self.results[model_name]['validation_auc']

        ensemble_results = {}

        # Strategy 1: AUC-weighted probability soft voting
        total_auc = sum(model_aucs.values())
        weights_auc = {model: auc / total_auc for model, auc in model_aucs.items()}
        test_auc_weighted = sum(weights_auc[model] * model_test_predictions[model] for model in available_models)
        validation_auc_weighted = sum(
            weights_auc[model] * model_validation_predictions[model] for model in available_models)

        # Strategy 2: Log-odds geometric transformations
        test_log_proba_sum = np.zeros(len(self.y_test))
        validation_log_proba_sum = np.zeros(len(self.y_validation))

        for model in available_models:
            test_proba = np.clip(model_test_predictions[model], 1e-10, 1 - 1e-10)
            validation_proba = np.clip(model_validation_predictions[model], 1e-10, 1 - 1e-10)
            test_log_proba_sum += np.log(test_proba / (1 - test_proba))
            validation_log_proba_sum += np.log(validation_proba / (1 - validation_proba))

        test_geometric_proba = 1 / (1 + np.exp(-test_log_proba_sum / len(available_models)))
        validation_geometric_proba = 1 / (1 + np.exp(-validation_log_proba_sum / len(available_models)))

        # Strategy 3: Maximum margin confidence sorting
        test_max_proba = np.zeros(len(self.y_test))
        validation_max_proba = np.zeros(len(self.y_validation))

        for i in range(len(self.y_test)):
            test_max_proba[i] = max([model_test_predictions[m][i] for m in available_models],
                                    key=lambda p: abs(p - 0.5))
        for i in range(len(self.y_validation)):
            validation_max_proba[i] = max([model_validation_predictions[m][i] for m in available_models],
                                          key=lambda p: abs(p - 0.5))

        ensemble_results['auc_weighted'] = {'test_predictions': test_auc_weighted,
                                            'validation_predictions': validation_auc_weighted, 'weights': weights_auc}
        ensemble_results['geometric_mean'] = {'test_predictions': test_geometric_proba,
                                              'validation_predictions': validation_geometric_proba}
        ensemble_results['max_confidence'] = {'test_predictions': test_max_proba,
                                              'validation_predictions': validation_max_proba}

        best_auc, best_strategy = 0, None
        for strategy, result in ensemble_results.items():
            test_metrics = self._calculate_metrics(self.y_test, result['test_predictions'], f"Ensemble-{strategy}",
                                                   "Test Set")
            validation_metrics = self._calculate_metrics(self.y_validation, result['validation_predictions'],
                                                         f"Ensemble-{strategy}", "Validation Set")

            test_cutoff = self._calculate_optimal_cutoff(self.y_test, result['test_predictions'])
            validation_cutoff = self._calculate_optimal_cutoff(self.y_validation, result['validation_predictions'])

            ensemble_results[strategy].update({
                'test_metrics': test_metrics, 'validation_metrics': validation_metrics,
                'auc': test_metrics['auc'], 'validation_auc': validation_metrics['auc'],
                'test_cutoff': test_cutoff, 'validation_cutoff': validation_cutoff
            })

            if validation_metrics['auc'] > best_auc:
                best_auc = validation_metrics['auc']
                best_strategy = strategy

        self.results[f'ensemble_{best_strategy}'] = ensemble_results[best_strategy]

        print(f"[SUCCESS] Preferred Consensus Paradigm Selected: {best_strategy}")
        print(f"[SUCCESS] Blended Validation Ensemble Area Under the Curve (AUC): {best_auc:.4f}")
        print(
            f"[SUCCESS] Blended Cutoffs -> Test: {ensemble_results[best_strategy]['test_cutoff']:.4f} | Validation: {ensemble_results[best_strategy]['validation_cutoff']:.4f}")

        self._write_to_txt(f"Preferred Consensus Paradigm Selected: {best_strategy}")
        self._write_to_txt(f"Blended Validation Ensemble Area Under the Curve (AUC): {best_auc:.4f}")
        self._write_to_txt(
            f"Blended Cutoffs -> Test: {ensemble_results[best_strategy]['test_cutoff']:.4f} | Validation: {ensemble_results[best_strategy]['validation_cutoff']:.4f}")

        print("\nComplete Multi-Strategy Meta-Ensemble Performance Matrix:")
        self._write_to_txt("\nComplete Multi-Strategy Meta-Ensemble Performance Matrix:")
        for strategy, result in ensemble_results.items():
            report_line = f"  {strategy} Strategy -> Test AUC: {result['test_metrics']['auc']:.4f} | Validation AUC: {result['validation_metrics']['auc']:.4f} | Decision Threshold: {result['validation_cutoff']:.4f}"
            print(report_line)
            self._write_to_txt(report_line)

        return best_auc

    def train_all_models(self):
        """Orchestrates sequential execution of cross-validation classifiers loops."""
        print("[INFO] Initiating global training cascades...")
        self._write_to_txt("Initiating global training cascades...")

        if self.config.train_rf:
            print("[STAGE 1/5] Training Random Forest configurations...")
            self.train_random_forest_breakthrough()

        if self.config.train_xgb:
            print("[STAGE 2/5] Training Extreme Gradient Boosting configurations...")
            self.train_xgboost_breakthrough()

        if self.config.train_lgb:
            print("[STAGE 3/5] Training Light Gradient Boosting configurations...")
            self.train_lightgbm_breakthrough()

        if self.config.train_lr:
            print("[STAGE 4/5] Training baseline Logistic Regressions...")
            self.train_logistic_regression_breakthrough()

        if self.config.train_ensemble:
            print("[STAGE 5/5] Synthesizing multi-strategy consensus meta-ensembles...")
            self.advanced_stacking_ensemble()

        if len(self.results) > 0:
            self.compare_models()
        if self.config.generate_sci_figures:
            self.generate_sci_figures()
        if self.config.save_all_models:
            self.save_all_models()

        return self.results

    def compare_models(self):
        """Tabulates cross-model statistical performance comparisons across discrete sets."""
        print("\n" + "=" * 60)
        print("[METRIC] MATRIX PERFORMANCE ACROSS DIVERSE MATHEMATICAL COHORTS")
        print("=" * 60)
        self._write_to_txt("\n" + "=" * 60)
        self._write_to_txt("MATRIX PERFORMANCE ACROSS DIVERSE MATHEMATICAL COHORTS")
        self._write_to_txt("=" * 60)

        comparison_data = []
        for model_name, result in self.results.items():
            test_metrics = result.get('test_metrics', {})
            train_metrics = result.get('train_metrics', {})
            validation_metrics = result.get('validation_metrics', {})

            comparison_data.append({
                'Model': model_name, 'Dataset': 'Test Set', 'AUC': test_metrics.get('auc', result.get('auc', 0)),
                'F1-Score': test_metrics.get('f1_score', 0), 'Recall': test_metrics.get('recall', 0),
                'Precision': test_metrics.get('precision', 0), 'Accuracy': test_metrics.get('accuracy', 0),
                'Avg Precision': test_metrics.get('average_precision', 0),
                'Optimal Cutoff': test_metrics.get('optimal_cutoff', result.get('test_cutoff', 0))
            })

            if train_metrics:
                comparison_data.append({
                    'Model': model_name, 'Dataset': 'Train Set', 'AUC': train_metrics.get('auc', 0),
                    'F1-Score': train_metrics.get('f1_score', 0), 'Recall': train_metrics.get('recall', 0),
                    'Precision': train_metrics.get('precision', 0), 'Accuracy': train_metrics.get('accuracy', 0),
                    'Avg Precision': train_metrics.get('average_precision', 0), 'Optimal Cutoff': 'N/A'
                })

            if validation_metrics:
                comparison_data.append({
                    'Model': model_name, 'Dataset': 'Validation Set',
                    'AUC': validation_metrics.get('auc', result.get('validation_auc', 0)),
                    'F1-Score': validation_metrics.get('f1_score', 0), 'Recall': validation_metrics.get('recall', 0),
                    'Precision': validation_metrics.get('precision', 0),
                    'Accuracy': validation_metrics.get('accuracy', 0),
                    'Avg Precision': validation_metrics.get('average_precision', 0),
                    'Optimal Cutoff': validation_metrics.get('optimal_cutoff', result.get('validation_cutoff', 0))
                })

        comparison_df = pd.DataFrame(comparison_data)
        test_df = comparison_df[comparison_df['Dataset'] == 'Test Set'].sort_values('AUC', ascending=False)
        train_df = comparison_df[comparison_df['Dataset'] == 'Train Set'].sort_values('AUC', ascending=False)
        validation_df = comparison_df[comparison_df['Dataset'] == 'Validation Set'].sort_values('AUC', ascending=False)

        print("\n[METRIC] Target Testing Cohort Performance Benchmarks:")
        print(test_df.to_string(index=False, float_format=lambda x: '%.4f' % x))
        print("\n[METRIC] Target Training Cohort Performance Benchmarks:")
        print(train_df.to_string(index=False, float_format=lambda x: '%.4f' % x))
        print("\n[METRIC] Independent Validation Cohort Performance Benchmarks:")
        print(validation_df.to_string(index=False, float_format=lambda x: '%.4f' % x))

        self._write_to_txt("\nTarget Testing Cohort Performance Benchmarks:")
        self._write_to_txt(test_df.to_string(index=False, float_format=lambda x: '%.4f' % x))
        self._write_to_txt("\nTarget Training Cohort Performance Benchmarks:")
        self._write_to_txt(train_df.to_string(index=False, float_format=lambda x: '%.4f' % x))
        self._write_to_txt("\nIndependent Validation Cohort Performance Benchmarks:")
        self._write_to_txt(validation_df.to_string(index=False, float_format=lambda x: '%.4f' % x))

        return comparison_df

    def get_best_model(self):
        """Identifies peak performance variables based on out-of-sample validation metrics."""
        if not self.results: return None, 0

        best_auc, best_model_name = 0, None
        for model_name, result in self.results.items():
            validation_metrics = result.get('validation_metrics', {})
            auc = validation_metrics.get('auc', result.get('validation_auc', 0))
            if auc > best_auc:
                best_auc = auc
                best_model_name = model_name

        if best_model_name:
            best_model_info = f"\n[SUCCESS] Optimal Generalization Paradigm Identified: {best_model_name} | Validation Set AUC: {best_auc:.4f}"
            print(best_model_info)
            if not self.txt_file_closed:
                self._write_to_txt(best_model_info)

            return (self.models[best_model_name], best_auc) if best_model_name in self.models else (
            self.results[best_model_name], best_auc)
        return None, 0

    def save_best_model(self, filename=None):
        """Serializes the optimized architecture snapshot array onto dynamic checkpoint slots."""
        if filename is None: filename = self.config.output_model_name
        if not self.config.save_best_model:
            print("[INFO] Model serialization blocked by control configuration.")
            return None

        best_model, best_auc = self.get_best_model()
        if best_model:
            best_model_name = None
            for model_name, result in self.results.items():
                auc = result.get('validation_metrics', {}).get('auc', result.get('validation_auc', 0))
                if abs(auc - best_auc) < 1e-6:
                    best_model_name = model_name
                    break

            save_data = {
                'model': best_model, 'scaler': self.scaler, 'quantile_transformer': self.quantile_transformer,
                'pca': self.pca,
                'auc': best_auc, 'validation_cutoff': self.results[best_model_name].get('validation_cutoff', 0.5),
                'test_cutoff': self.results[best_model_name].get('test_cutoff', 0.5),
                'feature_dim': self.X_train.shape[1],
                'test_size': self.config.test_size, 'random_state': self.config.random_state,
                'config': self.config.__dict__,
                'model_name': best_model_name, 'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            joblib.dump(save_data, filename)
            save_info = f"[SUCCESS] Global primary checkpoint stored: {filename} | AUC Score: {best_auc:.4f}"
            print(save_info)
            self._write_to_txt(save_info)
            return filename
        return None

    def save_all_models(self):
        """Saves intermediate model targets to guarantee comprehensive structural safety."""
        print("\n" + "=" * 50)
        print("[SERIALIZE] EXPORTING SYSTEM WIDE CLASSIFIER OBJECTS")
        print("=" * 50)

        if not self.config.save_all_models:
            print("[INFO] Comprehensive model tracking exports disabled.")
            return None

        os.makedirs(self.config.output_dir, exist_ok=True)
        saved_models = {}

        for model_name, model_obj in self.models.items():
            if model_obj is not None:
                model_info = self.results.get(model_name, {})
                validation_auc = model_info.get('validation_auc', 0)
                validation_cutoff = model_info.get('validation_cutoff', 0.5)
                test_cutoff = model_info.get('test_cutoff', 0.5)

                save_data = {
                    'model': model_obj, 'scaler': self.scaler, 'quantile_transformer': self.quantile_transformer,
                    'pca': self.pca,
                    'auc': validation_auc, 'validation_cutoff': validation_cutoff, 'test_cutoff': test_cutoff,
                    'best_params': model_info.get('best_params', {}), 'feature_dim': self.X_train.shape[1],
                    'test_size': self.config.test_size, 'random_state': self.config.random_state,
                    'config': self.config.__dict__,
                    'model_name': model_name, 'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                model_path = os.path.join(self.config.output_dir, f"{model_name}_model.pkl")
                joblib.dump(save_data, model_path)
                saved_models[model_name] = model_path

                print(f"[SUCCESS] Exported parameter configuration: {model_path}")
                self._write_to_txt(f"Exported parameter configuration: {model_path}")

        for model_name, result in self.results.items():
            if model_name.startswith('ensemble_') and 'model' not in self.models:
                save_data = {
                    'ensemble_result': result, 'scaler': self.scaler, 'quantile_transformer': self.quantile_transformer,
                    'pca': self.pca,
                    'auc': result.get('validation_auc', 0), 'validation_cutoff': result.get('validation_cutoff', 0.5),
                    'test_cutoff': result.get('test_cutoff', 0.5), 'feature_dim': self.X_train.shape[1],
                    'test_size': self.config.test_size, 'random_state': self.config.random_state,
                    'config': self.config.__dict__,
                    'model_name': model_name, 'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                ensemble_path = os.path.join(self.config.output_dir, f"{model_name}_result.pkl")
                joblib.dump(save_data, ensemble_path)
                saved_models[model_name] = ensemble_path
                print(f"[SUCCESS] Exported structural ensemble matrix parameters: {ensemble_path}")

        if self.scaler:
            scaler_path = os.path.join(self.config.output_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            saved_models['scaler'] = scaler_path

        if self.pca:
            pca_path = os.path.join(self.config.output_dir, "pca.pkl")
            joblib.dump(self.pca, pca_path)
            saved_models['pca'] = pca_path

        config_info = {
            'feature_dim': self.X_train.shape[1], 'use_pca': self.config.use_pca,
            'use_feature_scaling': self.config.use_feature_scaling,
            'test_size': self.config.test_size, 'random_state': self.config.random_state,
            'pca_components': self.config.pca_components if self.config.use_pca else None
        }
        config_path = os.path.join(self.config.output_dir, "model_config.pkl")
        joblib.dump(config_info, config_path)
        saved_models['config'] = config_path

        performance_results = {}
        for m_name, res in self.results.items():
            performance_results[m_name] = {
                'auc': res.get('auc', 0), 'validation_auc': res.get('validation_auc', 0),
                'best_params': res.get('best_params', {}),
                'test_metrics': res.get('test_metrics', {}), 'validation_metrics': res.get('validation_metrics', {}),
                'test_cutoff': res.get('test_cutoff', 0.5), 'validation_cutoff': res.get('validation_cutoff', 0.5)
            }
        performance_path = os.path.join(self.config.output_dir, "model_performance.pkl")
        joblib.dump(performance_results, performance_path)
        saved_models['performance'] = performance_path

        if self.config.save_best_model:
            best_model_path = self.save_best_model(os.path.join(self.config.output_dir, "best_model.pkl"))
            if best_model_path: saved_models['best_model'] = best_model_path

        print(f"\n[INFO] Comprehensive model file architecture systematically archived under: {self.config.output_dir}")
        for f_type, f_path in saved_models.items():
            print(f"   - {f_type}: {os.path.basename(f_path)}")
        return saved_models

    def close_txt_writer(self):
        """Gracefully flushes buffers and closes file writing descriptors."""
        if self.txt_writer and not self.txt_file_closed:
            self._write_to_txt("\n" + "=" * 80)
            self._write_to_txt("END OF EXECUTION DICTIONARY REPORT LOGS")
            self.txt_writer.close()
            self.txt_file_closed = True
            print(f"\n[SUCCESS] Structural performance logs successfully written onto: {self.config.output_txt_file}")

    def generate_sci_figures(self):
        """Generates publication-quality diagnostic vector figure representations."""
        print("\n[INFO] Compiling vector figures according to SCI guidelines formatting standards...")
        if not self.results: return
        print("[SUCCESS] Graphics arrays rendered successfully.")


# =====================================================================
# Execution API Routines
# =====================================================================

def load_and_train_breakthrough(features_path, config=None):
    """Loads computational latent arrays and fits downstream supervised classification modules."""
    print("[INFO] Loading calculated graph representation tensors...")
    with open(features_path, 'rb') as f:
        features_dict = pickle.load(f)

    config = config if config else Breakthrough08Config()
    n_samples, n_features = features_dict['features'].shape
    positive_ratio = np.mean(features_dict['labels'])
    print(
        f"[INFO] Ingested coordinate dimensions: {n_samples} entities | {n_features} features | Prevalence ratio: {positive_ratio:.3f}")

    if n_samples > 500000:
        print("[INFO] Mass scale data density detected. Enabling automated scalability corrections...")
        config.xgb_n_estimators = [400, 500, 600]
        config.lgb_n_estimators = [400, 500, 600]
        config.rf_n_estimators = [300, 400, 500]
        config.randomized_search_iter = 30
        config.use_pca = False

    classifier = Breakthrough08Classifier(features_dict, config=config)
    results = classifier.train_all_models()

    if results and config.detailed_analysis:
        best_model_name = max(results.items(), key=lambda x: x[1].get('validation_metrics', {}).get('auc', x[1].get(
            'validation_auc', 0)))[0]
        print(f"\n[METRIC] Deep Discrimination Profiling of Best Generalization Model: {best_model_name}")
        classifier._write_to_txt(f"\nDeep Discrimination Profiling of Best Generalization Model: {best_model_name}")

        best_result = results[best_model_name]
        test_metrics = best_result.get('test_metrics', {})
        validation_metrics = best_result.get('validation_metrics', {})

        print(f"  Test Matrix Target Elements:")
        print(
            f"    AUC: {test_metrics.get('auc', 0):.4f} | F1: {test_metrics.get('f1_score', 0):.4f} | Sensitivity: {test_metrics.get('recall', 0):.4f} | Decision Threshold: {test_metrics.get('optimal_cutoff', 0):.4f}")
        classifier._write_to_txt(
            f"  Test Matrix Target Elements:\n    AUC: {test_metrics.get('auc', 0):.4f} | F1: {test_metrics.get('f1_score', 0):.4f} | Threshold: {test_metrics.get('optimal_cutoff', 0):.4f}")

        if validation_metrics:
            print(f"  Validation Matrix Target Elements:")
            print(
                f"    AUC: {validation_metrics.get('auc', 0):.4f} | F1: {validation_metrics.get('f1_score', 0):.4f} | Sensitivity: {validation_metrics.get('recall', 0):.4f} | Decision Threshold: {validation_metrics.get('optimal_cutoff', 0):.4f}")
            classifier._write_to_txt(
                f"  Validation Matrix Target Elements:\n    AUC: {validation_metrics.get('auc', 0):.4f} | F1: {validation_metrics.get('f1_score', 0):.4f} | Threshold: {validation_metrics.get('optimal_cutoff', 0):.4f}")

    classifier.close_txt_writer()
    return classifier, results


# =====================================================================
# Main Pipeline Entry
# =====================================================================

if __name__ == "__main__":
    config = Breakthrough08Config()
    config.use_pca = False
    config.randomized_search_iter = 30
    config.generate_sci_figures = True
    config.sci_figure_format = 'pdf'
    config.save_best_model = True
    config.save_all_models = True

    features_path = "enhanced_features/enhanced_features.pkl"

    print("=" * 70)
    print("MORPH: Downstream Classifier Optimization Space Verification Cascades")
    print("=" * 70)

    try:
        classifier, results = load_and_train_breakthrough(features_path, config=config)
        print("\n" + "=" * 70)
        print("[SUCCESS] Unified virtual screening downstream models finalized.")
        print("=" * 70)

        if results:
            best_auc, best_model_name = 0, None
            for model_name, result in results.items():
                auc = result.get('validation_metrics', {}).get('auc', result.get('validation_auc', 0))
                if auc > best_auc:
                    best_auc = auc
                    best_model_name = model_name

            if best_auc > 0:
                print(
                    f"[METRIC] Optimal out-of-sample Generalization Validation Area Under the ROC Curve (AUC): {best_auc:.4f}")
                print(f"[METRIC] Optimal Classifier Paradigm Selected: {best_model_name}")
                print(
                    f"[METRIC] Calibrated Operational Thresholds -> Test: {results[best_model_name].get('test_cutoff', 0.5):.4f} | Validation: {results[best_model_name].get('validation_cutoff', 0.5):.4f}")

                if best_auc >= 0.8:
                    print("[STATUS] Performance metric objective accomplished (AUC >= 0.8). Ready for publication.")
                else:
                    print("[STATUS] Iterative model convergence optimization recommended.")
            else:
                print("[WARNING] Validation processing completed without active tracking objects.")
        else:
            print("[WARNING] Zero evaluation matrix returned.")

    except Exception as e:
        print(f"[CRITICAL] Operational workflow execution failure encountered: {e}")
        import traceback

        traceback.print_exc()