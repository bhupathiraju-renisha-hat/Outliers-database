import pandas as pd
import numpy as np
import json
import warnings
from sqlalchemy import create_engine, text
from sqlalchemy.types import DateTime, String, Integer
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def modified_zscore(series, threshold):
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index), "Skipped modified_zscore due to zero MAD"
    mzs = 0.6745 * (series - median) / mad
    return mzs.abs() > threshold, "Applied modified_zscore"

def iqr_outliers(series, multiplier):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound), "Applied IQR"

def isolation_forest_outliers(series):
    if len(series) < 20 or series.std() == 0:
        return pd.Series([False] * len(series), index=series.index), "Skipped isolationforest (low variance or too few rows)"
    clf = IsolationForest(contamination=0.1, random_state=42)
    reshaped = series.values.reshape(-1, 1)
    outliers = clf.fit_predict(reshaped)
    return pd.Series(outliers == -1, index=series.index), "Applied isolationforest"

def lof_outliers(series, n_neighbors):
    if len(series) < n_neighbors:
        return pd.Series([False] * len(series), index=series.index), "Skipped LOF: too few rows"
    if series.std() == 0 or series.nunique() <= 2:
        return pd.Series([False] * len(series), index=series.index), "Skipped LOF: zero std or few unique values"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LocalOutlierFactor(n_neighbors=n_neighbors)
            reshaped = series.values.reshape(-1, 1)
            outliers = clf.fit_predict(reshaped)
        return pd.Series(outliers == -1, index=series.index), "Applied LOF"
    except Exception as e:
        return pd.Series([False] * len(series), index=series.index), f"Failed LOF: {e}"

def rolling_window_outliers(series, window_size, threshold):
    rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, min_periods=1).std()
    z_scores = (series - rolling_mean) / rolling_std
    return z_scores.abs() > threshold, f"Applied {window_size}d_rollingwindow"

def detect_outliers(df, value_field, methods, base_thresholds):
    if value_field not in df.columns:
        return [(pd.Series([False] * len(df), index=df.index), method, "Skipped due to missing column") for method in methods]

    if df[value_field].isna().any():
        df[value_field] = df[value_field].fillna(0)

    thresholds = base_thresholds.get(value_field, base_thresholds.get("default", {}))
    results = []

    for method in methods:
        try:
            if method == "zscore":
                zscores = zscore(df[value_field])
                outliers = pd.Series(np.abs(zscores) > thresholds.get("zscore", 3), index=df.index)
                reason = "Applied zscore"
            elif method == "modified_zscore":
                outliers, reason = modified_zscore(df[value_field], thresholds.get("modified_zscore", 3.5))
            elif method == "iqr":
                outliers, reason = iqr_outliers(df[value_field], thresholds.get("iqr_multiplier", 1.5))
            elif method == "isolationforest":
                outliers, reason = isolation_forest_outliers(df[value_field])
            elif method == "lof":
                outliers, reason = lof_outliers(df[value_field], thresholds.get("lof_neighbors", 10))
            elif method == "7d_rollingwindow":
                outliers, reason = rolling_window_outliers(df[value_field], 7, thresholds.get("rolling_window", 3))
            elif method == "30d_rollingwindow":
                outliers, reason = rolling_window_outliers(df[value_field], 30, thresholds.get("rolling_window", 3))
            else:
                continue
            results.append((outliers, method, reason))
        except Exception as e:
            results.append((pd.Series([False] * len(df), index=df.index), method, f"Failed {method}: {e}"))
    return results

def main(config_path):
    config = load_config(config_path)
    dataset = config['datasets'][0]

    engine = create_engine(f"postgresql+psycopg2://{dataset['database']['user']}:{dataset['database']['password']}@{dataset['database']['host']}:{dataset['database']['port']}/{dataset['database']['database']}")

    query = f"""
    SELECT {', '.join(dataset['key_fields'] + dataset['value_fields'] + [dataset['datetime_field']])}
    FROM {dataset['table_name']}
    WHERE DATE({dataset['datetime_field']}) = %s
    """

    target_date = datetime.strptime(dataset['target_date'], dataset['timestamp_format'])
    df = pd.read_sql(query, engine, params=(target_date.strftime('%Y-%m-%d'),))
    df[dataset['datetime_field']] = pd.to_datetime(df[dataset['datetime_field']])

    all_results = []

    for value_field in dataset['value_fields']:
        temp_df = df.copy()
        for method in dataset['methods']:
            outliers, method_name, reason = detect_outliers(temp_df, value_field, [method], dataset['thresholds'])[0]
            temp_df[f"{value_field}_{method_name}_outlier"] = outliers
            logger.info(f"{method_name} for {value_field}: {reason}")

        def combine_methods(row):
            return [method for method in dataset['methods'] if row.get(f"{value_field}_{method}_outlier", False)]

        temp_df['outlier_type'] = temp_df.apply(combine_methods, axis=1)
        temp_df['total_outliers'] = temp_df['outlier_type'].apply(len)
        temp_df = temp_df[temp_df['total_outliers'] >= dataset.get('min_methods_flag', 3)]
        temp_df['outlier_type'] = temp_df['outlier_type'].apply(lambda x: f"{value_field}:{','.join(x)}")
        temp_df['value_field'] = value_field
        all_results.append(temp_df[['captured_at', 'source_db', 'schemaname', 'tablename', 'outlier_type', 'total_outliers', 'value_field']])

    if all_results:
        final_df = pd.concat(all_results)
        final_df = final_df.groupby(['captured_at', 'source_db', 'schemaname', 'tablename']).agg({
            'outlier_type': lambda x: ';'.join(x),
            'total_outliers': 'sum'
        }).reset_index()
        final_df = final_df.rename(columns={'total_outliers': 'total_outliers_detected'})
        final_df.to_sql('temp_outlier_results', engine, if_exists='replace', index=False, dtype={
            'captured_at': DateTime, 'source_db': String, 'schemaname': String,
            'tablename': String, 'outlier_type': String, 'total_outliers_detected': Integer
        })
        logger.info("Results written to temp_outlier_results.")
    else:
        logger.warning("No outlier results found for any value_field.")
    engine.dispose()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] != "--config":
        print("Usage: python outlier_detection.py --config <config_path>")
        sys.exit(1)
    main(sys.argv[2])