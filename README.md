# ShopSmart Retail Group — Data Preprocessing & Feature Engineering (Week 2)

**AnalystLab Africa · Machine Learning Internship · Week 2**
**Author:** Umar Muhammad

Turns nine raw ShopSmart tables into one clean, documented, leakage-aware, machine-learning-ready dataset for two planned models:

| Problem | Type | Target |
|---|---|---|
| How much is an order worth? | Regression | `Order_Value` |
| Will an order be returned? | Classification | `Is_Returned` |

This continues the Week 1 project (same dataset).

## Key findings
- **Complete but logically flawed.** No missing values or exact duplicates in any table, but the logic audit found 13 issues (e.g. returns recorded on cancelled orders, return dates before purchase dates, orders placed before the customer registered).
- **Leakage explained.** `Line_Total = Quantity x Unit_Price - Discount`, with `Discount` a 0-20 whole number subtracted as a flat amount. `Order_Value` is therefore almost pure arithmetic, which is why Week 1 showed R² = 1.0. Without the arithmetic inputs, R² is about 0.
- **Returns are not predictable from these fields.** 4.88% of orders are returned; cross-validated ROC-AUC is about 0.50 (chance). Accuracy is a misleading metric here; use recall, precision and ROC-AUC.
- **Outliers kept.** 410 high-value baskets (0.82%) are flagged by IQR but are genuine; RobustScaler limits their influence.

## Preprocessing pipeline
1. Inspect 9 tables (shape, types, missing, duplicates, statistics)
2. Logic audit (13 business-rule checks)
3. Clean: standardise text, parse dates, validate ranges, collapse returns to one flag per order, drop personal data
4. Engineer 14 features (time, basket value, margin, customer history, store)
5. Stratified 80/20 train/test split (seed 42) **before** fitting anything
6. Encode: ordinal (segment, age band), label (channel, gender), one-hot (5 nominal columns)
7. Scale (fitted on train only): Robust (skewed), MinMax (bounded), Standard (symmetric)
8. Detect outliers: IQR, Z-score, box plots
9. Select features: variance threshold, correlation pruning, leakage check, importance vs a random-noise probe (44 candidates -> 40 selected)
10. Save ML-ready and cleaned datasets, feature lists and fitted transformers

## Repository structure
```
ml-week2/
├── README.md
├── requirements.txt
├── notebook/ShopSmart_Week2_Preprocessing.ipynb
├── data/
│   ├── raw/          # the 9 original ShopSmart CSV files
│   └── processed/    # ShopSmart_ML_Ready.csv, ShopSmart_Cleaned.csv
├── models/           # feature_sets.json, preprocessing_artifacts.joblib
├── reports/          # Business_Understanding_Report, Data_Preprocessing_Report (.docx + .pdf), results.json
└── images/           # figures used in the notebook and reports
```

## Output files
| File | Shape | Description |
|---|---|---|
| `data/processed/ShopSmart_ML_Ready.csv` | 50,000 x 44 | `Order_ID`, `Split`, two targets, 40 encoded and scaled features |
| `data/processed/ShopSmart_Cleaned.csv` | 50,000 x 33 | Readable cleaned table, no personal data, before encoding/scaling |
| `models/feature_sets.json` | | Classification and regression feature lists, removed features, encodings |
| `models/preprocessing_artifacts.joblib` | | Fitted encoder and scalers for reuse |

**Using the feature sets:** use the `classification_Is_Returned` list for returns. For `Order_Value`, `Gross_Value` is excluded because it reconstructs the target; for a model that must predict *before* quantity and price are known, use `regression_Order_Value_no_arithmetic_inputs`. Filter on the `Split` column for the train/test partition.

## How to run
```bash
pip install -r requirements.txt
cd notebook
jupyter notebook ShopSmart_Week2_Preprocessing.ipynb
```
The notebook reads from `../data/raw` and writes to `../data/processed`, `../models`, `../images` and `../reports`.

## Reports
- `reports/Business_Understanding_Report` — business objective, stakeholders, data, implications
- `reports/Data_Preprocessing_Report` — every preprocessing decision with evidence

## Next steps (Week 3)
Train baseline models on the stored split, handle class imbalance for returns, compare against a majority-class baseline, and report the regression with and without arithmetic inputs. Confirm with the data owner whether `Discount` is meant to be a percentage.

#AnalystLabAfrica
