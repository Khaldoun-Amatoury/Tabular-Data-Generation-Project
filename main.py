from fastapi import FastAPI, HTTPException
from models.models import (get_available_datasets, setup, generate_data)
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import pandas as pd

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