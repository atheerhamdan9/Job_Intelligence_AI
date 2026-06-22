from fastapi import FastAPI
from fastapi import  HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from transformers import pipeline
import pandas as pd
import io

app = FastAPI(
    title="Job Intelligence AI"
)

MODEL_NAME = "lamisMohammed24/job-classifier-distilbert"

print("------- Loading DistilBERT model directly from Hugging Face Hub ------- ")

classifier = pipeline("text-classification", model=MODEL_NAME, truncation=True, max_length=512)
print("------- Model loaded successfully and API is live ------- ")

@app.get("/")
def root():
    return {"status": "Online", "connected_model": MODEL_NAME}

@app.post("/predict")
def predict_single(job_description: str = Form(..., description="paste job description")):
    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        cleaned_text = " ".join(job_description.split())

        prediction = classifier(cleaned_text)[0]
        return {
            "text_length": len(cleaned_text),
            "predicted_category": prediction["label"],
            "confidence_score": round(prediction["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch_csv")
async def predict_batch_csv(file: UploadFile = File(..., description="You must upload a CSV file containing the job_text column.")):

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if 'job_text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a column named exactly 'job_text'")

        jobs = df['job_text'].astype(str).tolist()
        cleaned_jobs = [" ".join(job.split()) for job in jobs]

        print(f" Processing batch of {len(cleaned_jobs)} jobs via API...")

        predictions = classifier(cleaned_jobs, batch_size=16)

        df['predicted_category'] = [p['label'] for p in predictions]
        df['confidence_score'] = [round(p['score'], 4) for p in predictions]

        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")

        response.headers["Content-Disposition"] = f"attachment; filename=classified_{file.filename}"
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
