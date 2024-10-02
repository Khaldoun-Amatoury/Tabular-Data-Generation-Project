from fastapi import FastAPI, HTTPException
from models.models import setup, generate_data
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import os
from datetime import datetime
from evaluation.evaluation import (
    evaluate_data_quality,
    run_diagnostic_sdv,
    quality_report_sdmetrics,
    run_syntheval,
    get_column_plot_sdv
)

app = FastAPI()

class GenerateDataRequest(BaseModel):
    model_name: str
    num_rows: int = 10

def get_latest_dataset():
    datasets_folder = os.path.join(os.getcwd(), 'datasets')
    files = os.listdir(datasets_folder)
    if not files:
        raise FileNotFoundError("No datasets found")
    latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(datasets_folder, f)))
    return os.path.join(datasets_folder, latest_file)

@app.post("/generate-data")
async def generate_synthetic_data(request: GenerateDataRequest):
    try:
        data, metadata = setup()
        synthetic_data = generate_data(request.model_name, data, metadata, request.num_rows, save_data=False)

        # Create filename with model name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.model_name}_{timestamp}.csv"
        
        # Save to datasets folder
        save_path = os.path.join(os.getcwd(), 'datasets', filename)
        synthetic_data.to_csv(save_path, index=False)

        return JSONResponse(content={"message": f"Data generated and saved as {filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-latest")
async def evaluate_latest_dataset():
    try:
        # Get the latest dataset
        latest_file_path = get_latest_dataset()
        synthetic_data = pd.read_csv(latest_file_path)

        # Get original data and metadata
        data, metadata = setup()

        # Perform evaluations
        quality_report = evaluate_data_quality(data, synthetic_data, metadata)
        diagnostic_report = run_diagnostic_sdv(data, synthetic_data, metadata)
        syntheval_score = run_syntheval(data, synthetic_data, metadata)
        quality_report_sdm = quality_report_sdmetrics(data, synthetic_data, metadata)

        # Prepare response
        evaluation_results = {
            "file_evaluated": os.path.basename(latest_file_path),
            "quality_report": quality_report,
            "diagnostic_report": diagnostic_report,
            "syntheval_score": syntheval_score,
            "quality_report_sdmetrics": quality_report_sdm
        }

        return JSONResponse(content=evaluation_results)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)