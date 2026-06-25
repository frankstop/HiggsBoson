# Higgs Boson Classification Summary

## Dataset and preprocessing

The dataset contains 250,000 rows and 30 model features.
The target is binary: background is encoded as 0 and Higgs signal is encoded as 1.
Class counts were 164,333 background events and 85,667 signal events.
The minority-to-majority ratio was 0.521. Random oversampling was applied only inside the training folds.

The Kaggle Higgs data uses -999 as a sentinel for undefined kinematic quantities. Those values were converted to missing values and imputed with the median within each training split. Median imputation was selected because these variables are continuous and can contain skewed tails. Standard scaling was applied after imputation for every model so logistic regression received well-conditioned inputs.

There were no categorical predictors, so no feature encoding was required.

## Model comparison

Three supervised models were trained: logistic regression, random forest, and XGBoost gradient boosting.

| model               |   precision |   recall |     f1 |   roc_auc |
|:--------------------|------------:|---------:|-------:|----------:|
| XGBoost             |      0.7078 |   0.8257 | 0.7622 |    0.9063 |
| Random Forest       |      0.7654 |   0.758  | 0.7617 |    0.9049 |
| Logistic Regression |      0.5884 |   0.76   | 0.6633 |    0.8133 |

The strongest baseline model by ROC-AUC was XGBoost with ROC-AUC 0.9063.

Cross-validation was used to check model stability:

| model               |   precision_mean |   precision_std |   recall_mean |   recall_std |   f1_mean |   f1_std |   roc_auc_mean |   roc_auc_std |
|:--------------------|-----------------:|----------------:|--------------:|-------------:|----------:|---------:|---------------:|--------------:|
| XGBoost             |           0.711  |          0.0011 |        0.8239 |       0.0011 |    0.7633 |   0.0004 |         0.9073 |        0.0004 |
| Random Forest       |           0.7655 |          0.0014 |        0.7573 |       0.0008 |    0.7614 |   0.0006 |         0.906  |        0.0006 |
| Logistic Regression |           0.5893 |          0.0013 |        0.7674 |       0.0031 |    0.6666 |   0.0007 |         0.816  |        0.0004 |

## Hyperparameter tuning

The selected model for tuning was XGBClassifier.
Best cross-validated ROC-AUC during grid search: 0.9100.
Hold-out ROC-AUC after tuning: 0.9093.
Best parameters: `{"model__learning_rate": 0.08, "model__max_depth": 5, "model__n_estimators": 350, "model__subsample": 0.85}`.

Tuning was assessed with ROC-AUC because the task is a signal-vs-background ranking problem and accuracy alone can hide poor signal recall.

## Top features

Permutation importance on the hold-out set identified the three strongest drivers:

| feature                     |   importance_mean |   importance_std |
|:----------------------------|------------------:|-----------------:|
| DER_mass_MMC                |           0.09661 |          0.00334 |
| DER_mass_transverse_met_lep |           0.03572 |          0.0029  |
| DER_mass_vis                |           0.03297 |          0.00175 |

Higher permutation importance means shuffling that feature caused a larger drop in ROC-AUC, so the model depended on that variable more heavily for separating signal from background.

## Generated files

- `metrics_baseline.csv`
- `cross_validation.csv`
- `best_model_grid_search.json`
- `top_features.csv`
- `target_distribution.png`
- `roc_curves.png`
- `precision_recall_curves.png`
- `top_features.png`
