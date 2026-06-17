from fastapi import FastAPI

app = FastAPI(
    title="Job Intelligence AI",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Job Intelligence AI API"}
