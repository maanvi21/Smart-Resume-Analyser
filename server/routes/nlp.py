# File for nlp-module endpoint
from fastapi import APIRouter, HTTPException
import json
import spacy
import pytextrank
from pathlib import Path

router = APIRouter()

@router.get('/nlp-module')
async def nlp_module():
    try:
        file_path = Path("uploaded_resume.json").absolute()
        output_file = Path("nlp_results.json").absolute()
        
        print(f"Starting NLP processing")
        print(f"Reading from: {file_path}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Resume data file not found")
        
        with open(file_path, "r", encoding="utf-8") as file:
            resume_data = json.load(file)
        
        if not resume_data.get("text"):
            raise HTTPException(status_code=400, detail="No resume text found in data")
        
        resume_text = resume_data.get("text", "")
        resume_keywords_requirements = resume_data.get("requirements", [])
        
        # Make sure resume text isn't empty
        if not resume_text:
            raise HTTPException(status_code=400, detail="No text content found in the resume")
        
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            # If model not found, try to download it
            from spacy.cli import download
            download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
        
        # Add PyTextRank to pipeline
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe("textrank")
        
        # Process text with spaCy
        nlp_text = nlp(resume_text)
        
        # Extract keywords using PyTextRank
        ranked_phrases = [phrase.text for phrase in nlp_text._.phrases][:20]  # Limit to top 20 phrases
        
        # Extract noun chunks and entities
        noun_keywords = set(chunk.text.lower() for chunk in nlp_text.noun_chunks)
        ner_keywords = set(ent.text.lower() for ent in nlp_text.ents)
        
        # Convert requirements to lowercase for case-insensitive matching
        requirements_lower = set(req.lower() for req in resume_keywords_requirements)
        
        # Find matching keywords
        matching_noun_keywords = noun_keywords.intersection(requirements_lower)
        matching_ner_keywords = ner_keywords.intersection(requirements_lower)
        
        # Calculate match percentage
        total_req = len(requirements_lower)
        matched_req = len(set(matching_noun_keywords).union(matching_ner_keywords))
        match_percentage = int((matched_req / total_req * 100)) if total_req > 0 else 0
        
        # Determine if suitable (e.g., more than 70% match)
        suitable = match_percentage >= 70
        
        # Create result dictionary
        results = {
            "ranked_phrases": ranked_phrases,
            "noun_keywords": list(noun_keywords),
            "ner_keywords": list(ner_keywords),
            "matching_noun_keywords": list(matching_noun_keywords),
            "matching_ner_keywords": list(matching_ner_keywords),
            "matchPercentage": match_percentage,
            "suitable": suitable
        }
        
        # Save results to a JSON file
        print(f"Writing results to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(results, json_file, indent=4)
        
        return results
        
    except Exception as e:
        print(f"Error in NLP processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))