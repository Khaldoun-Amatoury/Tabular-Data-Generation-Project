from sdv.evaluation.single_table import evaluate_quality,run_diagnostic,get_column_plot
from syntheval import SynthEval
from sdmetrics.reports.single_table import QualityReport


def run_diagnostic(real_data, synthetic_data, metadata):
    diagnostic_report = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata)
    return diagnostic_report


def evaluate_quality(real_data, synthetic_data, metadata):
    quality_report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata)
    return quality_report


def get_column_plot(real_data, synthetic_data, metadata, column_name):
    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=column_name
    )
    return fig


def run_syntheval(real_data, synthetic_data, metadata):
    evaluator = SynthEval(real_data)
    score = evaluator.evaluate(synthetic_data, presets_file = "full_eval")
    return score 


def quality_report_sdmetrics(real_data, synthetic_data, metadata):
    quality_report = QualityReport()
    score = quality_report.generate(real_data, synthetic_data, metadata.to_dict())
    return score
