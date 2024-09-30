from fastapi import FastAPI, HTTPException
from models.models import (get_available_datasets, setup, generate_data)
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import pandas as pd
from evaluation.evaluation import (evaluate_data_quality, run_diagnostic_sdv, quality_report_sdmetrics, run_syntheval, get_column_plot_sdv)

app = FastAPI()

class GenerateDataRequest(BaseModel):
    model_name: str
    num_rows: int = 10
    save_data: bool = False


@app.get("/available-datasets")
async def available_datasets():
    try:
        datasets = get_available_datasets()
        
        if isinstance(datasets, pd.DataFrame):
            datasets_json = datasets.to_dict(orient="records")
        else:
            datasets_json = [str(dataset) for dataset in datasets]  

        return {"datasets": datasets_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-data")
async def generate_synthetic_data(request: GenerateDataRequest):
    try:
        data, metadata = setup()  
        synthetic_data = generate_data(request.model_name, data, metadata, request.num_rows, request.save_data)

        buffer = io.StringIO()
        synthetic_data.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-quality")
async def evaluate_quality():
    try:
        data, metadata = setup()
        synthetic_data = generate_data("gaussian_copula", data, metadata, num_rows=100) 
        quality_report = evaluate_data_quality(data, synthetic_data, metadata)
        return {"quality_report": quality_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-diagnostic")
async def run_diagnostic_endpoint():
    try:
        data, metadata = setup()
        synthetic_data = generate_data("gaussian_copula", data, metadata, num_rows=100) 
        diagnostic_report = run_diagnostic_sdv(data, synthetic_data, metadata)
        return {"diagnostic_report": diagnostic_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-column-plot")
async def get_column_plot_endpoint(column_name: str):
    try:
        data, metadata = setup()
        synthetic_data = generate_data("gaussian_copula", data, metadata, num_rows=100) 
        fig = get_column_plot_sdv(data, synthetic_data, metadata, column_name)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="image/png", headers={"Content-Disposition": f"inline; filename={column_name}_plot.png"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-syntheval")
async def run_syntheval_endpoint():
    try:
        data, metadata = setup()
        synthetic_data = generate_data("gaussian_copula", data, metadata, num_rows=100)  
        score = run_syntheval(data, synthetic_data, metadata)
        return {"syntheval_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quality-report")
async def quality_report_endpoint():
    try:
        data, metadata = setup()
        synthetic_data = generate_data("gaussian_copula", data, metadata, num_rows=100)
        quality_report = quality_report_sdmetrics(data, synthetic_data, metadata)
        return {"quality_report_sdmetrics": quality_report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)