from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your routers
from routes.resume import router as extract_router
from routes.nlp import router as nlp_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(extract_router)
app.include_router(nlp_router)

@app.get("/")
def home():
    return {"message": "Smart Resume Analyzer API is running!"}