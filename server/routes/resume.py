# File for extract-features endpoint
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import List
import json
import pdfplumber

router = APIRouter()

@router.post("/extract-features")
async def extract_features(
    file: UploadFile = File(...),
    requirements: str = Form(...)
):
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")
        
        # Process the PDF content to extract text
        extracted_text = ""
        with pdfplumber.open(file.file) as pdf:
            extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        requirements_list = [req.strip().lower() for req in requirements.split(",")]
        
        data = {
            "filename": file.filename, 
            "text": extracted_text, 
            "requirements": requirements_list
        }
        
        with open("uploaded_resume.json", "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
        
        return {"message": "Resume processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))