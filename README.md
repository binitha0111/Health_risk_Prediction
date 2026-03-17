# Healthcare Risk Prediction

Patient readmission risk prediction using Python, Databricks, and Power BI.

## Project Summary
- **Dataset**: This project uses the Diabetes 130-US hospitals dataset from Kaggle:

      https://www.kaggle.com/datasets/brandao/diabetes

- **Best Model**: Gradient Boosting (AUC: 0.6616)
- **Patients Scored**: 71,515 unique patients
- **High Risk Patients**: 3,420 flagged for intervention

## Tech Stack
- **Python** — data cleaning, feature engineering, ML modeling
- **Databricks** — Bronze/Silver/Gold Delta Lake pipeline
- **Databricks SQL** — analytical queries on Delta tables
- **Scikit-learn** — Logistic Regression, Random Forest, Gradient Boosting
- **Power BI** — 4 interactive dashboard pages connected to Databricks

## Project Structure
```
Health_risk_Prediction/
├── dashboards/
│   ├── health_risk_prediction.pbix          # Power BI (local CSV)
│   └── health_risk_prediction_databricks.pbix # Power BI (Databricks live)
├── models/
│   ├── best_model.pkl                       # Saved Gradient Boosting model
│   ├── feature_columns.json                 # Feature list for scoring
│   ├── metrics.json                         # Model performance metrics
│   ├── feature_importance.png               # Feature importance chart
│   └── roc_curve.png                        # ROC curve chart
├── notebooks/
│   ├── Databricks/
│   │   ├── Pipelines/
│   │   │   ├── 01_silver_cleaning.ipynb     # Data cleaning on Databricks
│   │   │   ├── 02_gold_layer.ipynb          # Feature engineering on Databricks
│   │   │   └── 03_scoring_pipeline.ipynb    # Daily risk scoring on Databricks
│   │   └── SQL/
│   │       ├── high_risk_indicators         # SQL - insulin & utilizer analysis
│   │       ├── readmission_by_age           # SQL - readmission rate by age
│   │       └── risk_summary_by_readmission  # SQL - risk summary by status
│   └── Local/
│       ├── 01_data_exploration.ipynb        # EDA and data cleaning
│       └── 02_scoring_pipeline.py           # Local daily scoring pipeline
└── README.md
```
## Pipeline Architecture
```
Raw Data (Kaggle CSV)
      ↓
Bronze Layer (diabetic_data Delta table - Databricks)
      ↓
Silver Layer (cleaned, deduplicated - 71,515 patients - Databricks)
      ↓
Gold Layer (feature engineered - 54 columns - Databricks)
      ↓
ML Model (Gradient Boosting retrained in Databricks - AUC 0.6364)
      ↓
Daily Risk Scores (automated Databricks Job - 6 AM)
      ↓
Power BI Dashboard (4 pages - live Databricks connection)
```

## Model Results
| Model | AUC |
|---|---|
| Logistic Regression | 0.6137 |
| Random Forest | 0.6361 |
| Balanced Random Forest | 0.6550 |
| Gradient Boosting | 0.6616 |

## Key Features
| Feature | Description |
|---|---|
| num_active_meds | Count of active medications |
| on_insulin | Insulin prescription flag |
| total_prior_visits | Total previous hospital visits |
| high_utilizer | 3+ prior visits flag |
| med_changed | Medication changed during visit |
| age_numeric | Age range converted to midpoint |

## How to Run Locally
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run notebooks in order
python notebooks/01_data_exploration.ipynb
python notebooks/02_scoring_pipeline.py
```

## Databricks Setup
1. Upload `diabetic_data.csv` via Data Ingestion
2. Run `01_silver_cleaning` notebook
3. Run `02_gold_layer` notebook
4. Run `03_scoring_pipeline` notebook
5. Schedule `03_scoring_pipeline` as daily Job (6 AM)
