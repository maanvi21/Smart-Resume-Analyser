from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes.resume import router as extract_router
from routes.nlp import router as nlp_router

app = FastAPI(
    title="Smart Resume Analyzer API",
    description=(
        "AI-powered resume analysis with HuggingFace models "
        "and dataset-calibrated category prediction."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router)
app.include_router(nlp_router)


@app.get("/")
def home():
    return {"message": "Smart Resume Analyzer API is running!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)