from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from models.models import generate_data
from evaluation.evaluation import (
    run_diagnostic, evaluate_quality, 
    get_column_plot, run_syntheval, 
    quality_report_sdmetrics
)

app = FastAPI()

# Pydantic models for API request validation
class GenerateDataRequest(BaseModel):
    model_name: str
    num_rows: int = 10
    save_data: bool = False
    data: list  # Expecting data in list of dictionaries
    metadata: dict  # Metadata as a dictionary

class EvaluateRequest(BaseModel):
    real_data: list
    synthetic_data: list
    metadata: dict
    column_name: str = None  # Optional for column-specific evaluation

# Synthetic Data Generation API
@app.post("/generate-data/")
def generate_synthetic_data(request: GenerateDataRequest):
    try:
        # Convert list of dict to pandas DataFrame
        data = pd.DataFrame(request.data)
        metadata = request.metadata
        
        # Call the data generation function with user inputs
        synthetic_data = generate_data(
            model_name=request.model_name,
            data=data,
            metadata=metadata,
            num_rows=request.num_rows,
            save_data=request.save_data
        )

        # Return the generated synthetic data as a dictionary
        return synthetic_data.to_dict(orient="records")
    except ValueError as e:
        # Handle potential value errors and return an HTTP 400 error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any other exceptions and return an HTTP 500 error
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# API to Evaluate Synthetic Data Quality
@app.post("/evaluate-quality/")
def evaluate_synthetic_quality(request: EvaluateRequest):
    try:
        # Convert real and synthetic data from list of dicts to DataFrames
        real_df = pd.DataFrame(request.real_data)
        synthetic_df = pd.DataFrame(request.synthetic_data)
        metadata = request.metadata

        # Evaluate the quality of the synthetic data
        quality_report = evaluate_quality(real_df, synthetic_df, metadata)
        return quality_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in evaluating quality: {str(e)}")


# API to Run Diagnostic on Synthetic Data
@app.post("/run-diagnostic/")
def run_synthetic_diagnostic(request: EvaluateRequest):
    try:
        real_df = pd.DataFrame(request.real_data)
        synthetic_df = pd.DataFrame(request.synthetic_data)
        metadata = request.metadata
        
        # Run diagnostic evaluation
        diagnostic_report = run_diagnostic(real_df, synthetic_df, metadata)
        return diagnostic_report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in running diagnostic: {str(e)}")


# API to Plot Column Comparison between Real and Synthetic Data
@app.post("/get-column-plot/")
def get_column_plot_api(request: EvaluateRequest):
    try:
        real_df = pd.DataFrame(request.real_data)
        synthetic_df = pd.DataFrame(request.synthetic_data)
        metadata = request.metadata
        
        # Check if column name is provided
        if not request.column_name:
            raise HTTPException(status_code=400, detail="Column name is required.")

        # Generate the column plot
        plot = get_column_plot(real_df, synthetic_df, metadata, request.column_name)
        return plot
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in generating column plot: {str(e)}")


# API to Run Syntheval on Synthetic Data
@app.post("/run-syntheval/")
def run_syntheval_api(request: EvaluateRequest):
    try:
        real_df = pd.DataFrame(request.real_data)
        synthetic_df = pd.DataFrame(request.synthetic_data)
        metadata = request.metadata
        
        # Run Syntheval evaluation
        score = run_syntheval(real_df, synthetic_df, metadata)
        return score
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in running Syntheval: {str(e)}")


# API to Generate Quality Report using SDMetrics
@app.post("/quality-report-sdmetrics/")
def quality_report_sdmetrics_api(request: EvaluateRequest):
    try:
        real_df = pd.DataFrame(request.real_data)
        synthetic_df = pd.DataFrame(request.synthetic_data)
        metadata = request.metadata
        
        # Run SDMetrics evaluation
        score = quality_report_sdmetrics(real_df, synthetic_df, metadata)
        return score
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in running SDMetrics report: {str(e)}")


# Start the FastAPI app
# You can run the FastAPI app using the following command:
# uvicorn main:app --reload
