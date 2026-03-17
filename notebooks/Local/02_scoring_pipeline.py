import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# ── Load saved model and features ──────────────────────────
with open('models/best_model.pkl', 'rb') as f: #gradient boost model in data exploration file
    model = pickle.load(f)

with open('models/feature_columns.json') as f: #important fetures in data exploration file
    feature_cols = json.load(f)

print("✅ Model loaded")
print(f"   Features expected: {len(feature_cols)}")

# ── Load and prepare data ───────────────────────────────────
df = pd.read_csv('data/raw/diabetic_data.csv')

# Apply same cleaning as before
df = df.drop(columns=['weight', 'payer_code', 'medical_specialty'])
df = df[df['gender'] != 'Unknown/Invalid']
df = df.sort_values('encounter_id')
df = df.drop_duplicates(subset='patient_nbr', keep='first')

# Replace missing values
df['race'] = df['race'].replace('?', 'Unknown')
df['diag_1'] = df['diag_1'].replace('?', 'Unknown')
df['diag_2'] = df['diag_2'].replace('?', 'Unknown')
df['diag_3'] = df['diag_3'].replace('?', 'Unknown')

# Feature engineering
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone']

df['num_active_meds'] = (df[med_cols] != 'No').sum(axis=1) #same feautre we created in data exploration file
df['on_insulin'] = (df['insulin'] != 'No').astype(int)
df['med_changed'] = (df['change'] == 'Ch').astype(int)
df['total_prior_visits'] = (df['number_outpatient'] +
                            df['number_emergency'] +
                            df['number_inpatient'])
df['high_utilizer'] = (df['total_prior_visits'] >= 3).astype(int)

age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_map)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['race', 'gender', 'max_glu_serum', 'A1Cresult']:
    df[col] = le.fit_transform(df[col])

# ── Score patients ──────────────────────────────────────────
X = df[feature_cols]
df['risk_score'] = (model.predict_proba(X)[:, 1] * 100).round(1) # predict_proba returns probability(btw 0 and 1) of each patient being readmitted 
df['risk_tier'] = pd.cut(df['risk_score'], # divides risk_score into 4
    bins=[0, 30, 60, 80, 100],
    labels=['Low', 'Medium', 'High', 'Critical'])

# ── Risk tier distribution ──────────────────────────────────
print("\nRisk Tier Distribution:")
for tier in ['Critical', 'High', 'Medium', 'Low']:
    n = (df['risk_tier'] == tier).sum()
    pct = n / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {tier:>8}: {n:>6,} ({pct:4.1f}%) {bar}")

# ── Save outputs ────────────────────────────────────────────
output = df[['patient_nbr', 'age', 'race', 'gender',
             'time_in_hospital', 'num_lab_procedures',
             'num_medications', 'risk_score', 'risk_tier']].copy()

output.to_csv('data/processed/daily_risk_scores.csv', index=False) #goes to powerBI

# High risk alert list
alerts = output[output['risk_tier'].isin(['Critical', 'High'])] \
    .sort_values('risk_score', ascending=False)
alerts.to_csv('data/processed/high_risk_alerts.csv', index=False) #only Critical and High patients sorted by risk score → goes to clinical team

print(f"\n✅ Scored {len(df):,} patients")
print(f"✅ Saved daily_risk_scores.csv")
print(f"✅ High risk alerts: {len(alerts):,} patients")
print(f"   Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
