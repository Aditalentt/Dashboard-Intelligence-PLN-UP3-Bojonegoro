import pandas as pd

def load_data():
    df = pd.read_excel('sorek cuy.xlsx')

    # feature dasar
    df['TOTAL_KWH'] = df['KWHLWBP'] + df['KWHWBP'] + df['BLOK3']
    df['TOTAL_RP'] = df['RPTAG']

    return df

def aggregate_unit(df):
    unit = df.groupby('UNITUP').agg(
        TOTAL_KWH=('TOTAL_KWH', 'sum'),
        TOTAL_RP=('TOTAL_RP', 'sum'),
        JUMLAH_DATA=('TOTAL_KWH', 'count')
    ).reset_index()

    unit['AVG_KWH'] = unit['TOTAL_KWH'] / unit['JUMLAH_DATA']
    unit['AVG_RP'] = unit['TOTAL_RP'] / unit['JUMLAH_DATA']

    return unit

def add_segmentation(df):
    df['RP_PER_KWH'] = df['TOTAL_RP'] / df['TOTAL_KWH'].replace(0, 1)

    df['score_kwh'] = pd.qcut(df['TOTAL_KWH'], 3, labels=[1,2,3], duplicates = 'drop')
    df['score_rp'] = pd.qcut(df['RP_PER_KWH'], 3, labels=[1,2,3], duplicates = 'drop')

    df['score_total'] = df['score_kwh'].astype(int) + df['score_rp'].astype(int)

    return df

def segment(row):
    if row['score_total'] >= 5:
        return 'High Value'
    elif row['score_kwh'] == 3:
        return 'High Consumption'
    elif row['score_rp'] == 3:
        return 'High Tarif'
    else:
        return 'Low Value'

def apply_segmentation(df):
    df['SEGMENT'] = df.apply(segment, axis = 1)
    return df

def segment_action(seg):
    if seg == 'High Value':
        return 'Pertahankan dan Loyalty Program'
    elif seg == 'High Consumption':
        return 'Optimasi tarif/efisiensi'
    elif seg == 'High Tarif':
        return 'Edukasi'
    else:
        return 'Upsell konsumsi'
    
def apply_action(df):
    df['ACTION'] = df['SEGMENT'].apply(segment_action)
    return df

def get_segment_comparison(df, filtered):
    global_dist = df['SEGMENT'].value_counts(normalize=True)
    local_dist = filtered['SEGMENT'].value_counts(normalize=True)

    compare = pd.DataFrame({
        'Global': global_dist,
        'Local': local_dist
    }).fillna(0)

    return compare
    
def segment_distribution(df):

    result = {}

    for seg in df['SEGMENT'].unique():
        seg_df = df[df['SEGMENT'] == seg]
        summary = (seg_df.groupby('TARIP').agg(Total_KWH=('TOTAL_KWH','sum'), Total_Tagihan=('TOTAL_RP','sum')).reset_index())
        total_tagihan = summary['Total_Tagihan'].sum()

        summary['Kontribusi_Tagihan (%)'] = (summary['Total_Tagihan']/ total_tagihan * 100).round(2)
        result[seg] = summary.sort_values('Total_Tagihan', ascending=False)

    return result

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Predictive Model
def train_model(df):

    le = LabelEncoder()
    df['TARIP_ENC'] = le.fit_transform(df['TARIP'])
    features = ['TOTAL_KWH', 'DAYA', 'JAMNYALA', 'TARIP_ENC']
    
    df_model = df.dropna(subset = features + ['TOTAL_RP']).copy()

    X = df_model[features]
    y = df_model['TOTAL_RP']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, 1))) * 100

    df_model['PRED_RP'] = model.predict(X)

    return df_model, model, mae, mape

# Anomaly Detection
def detect_anomaly(df):
    q1 = df['TOTAL_RP'].quantile(0.25)
    q3 = df['TOTAL_RP'].quantile(0.75)
    iqr = q3 - q1

    df['ANOMALI_TAGIHAN'] = ((df['TOTAL_RP'] < q1 - 1.5*iqr) | (df['TOTAL_RP'] > q3 + 1.5*iqr))
    df['ANOMALI_JAM'] = (df['JAMNYALA'] > 720)
    return df

# Preprocessing & Tokenizing
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Cleaning Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Tokenizing
def tokenize(text):
    return text.split()

def train_nlp_model(df_keluhan):

    df_nlp = df_keluhan.copy()

    df_nlp['clean_text'] = df_nlp['KELUHAN'].apply(clean_text)

    X = df_nlp['clean_text']
    y = df_nlp['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    tfidf = TfidfVectorizer()

    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return accuracy, report, y_test, y_pred