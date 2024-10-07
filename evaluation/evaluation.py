from sdv.evaluation.single_table import evaluate_quality,run_diagnostic,get_column_plot
from syntheval import SynthEval
from sdmetrics.reports.single_table import QualityReport
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, roc_auc_score
import numpy as np 

def evaluate_data_sklearn(real_data, synthetic_data):
    precision = precision_score(real_data, synthetic_data, average='weighted')
    recall = recall_score(real_data, synthetic_data, average='weighted')
    f1 = f1_score(real_data, synthetic_data, average='weighted')
    mae = mean_absolute_error(real_data, synthetic_data)
    rmse = np.sqrt(mean_squared_error(real_data, synthetic_data))
    r2 = r2_score(real_data, synthetic_data)
    silhouette = silhouette_score(real_data, synthetic_data)
    auc = roc_auc_score(real_data, synthetic_data)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'silhouette': silhouette,
        'auc': auc
    }

def evaluate_data_quality(real_data, synthetic_data, metadata):
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata)
    return quality_report


def run_diagnostic_sdv(real_data, synthetic_data, metadata):
    diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata)
    return diagnostic_report


def get_column_plot_sdv(real_data, synthetic_data, metadata, column_name):
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=column_name
    )
    return fig


def run_syntheval(real_data, synthetic_data, metadata):
    real_data.dropna(inplace=True)
    synthetic_data.dropna(inplace=True)
    evaluator = SynthEval(real_data)
    score = evaluator.evaluate(synthetic_data, presets_file = "full_eval")
    return score 


def quality_report_sdmetrics(real_data, synthetic_data, metadata):
    quality_report = QualityReport()
    score = quality_report.generate(real_data, synthetic_data, metadata.to_dict())
    return score


