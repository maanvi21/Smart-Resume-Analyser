import React, { useState } from 'react';
import './ResumeParser.css';

function ResumeParser() {
  const [file, setFile] = useState(null);
  const [requirements, setRequirements] = useState('');
  const [results, setResults] = useState(null);
//   wait time until the results are shown. 
// if loaded, show results container
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');
    } else {
      setFile(null);
      setError('Please upload a PDF file');
    }
  };

  const handleRequirementsChange = (e) => {
    setRequirements(e.target.value);
   
    
  };


//   here keywords will be sent to backend api & the file will be sent to backend api 
const handleSubmit = async (e) => {
  e.preventDefault();
  
  if (!file || !requirements.trim()) {
    setError('Please upload a resume PDF and enter job requirements');
    return;
  }
  
  setIsLoading(true);
  
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('requirements', requirements);

    // First API call to extract features
    const extractResponse = await fetch('http://localhost:8000/extract-features', {
      method: 'POST',
      body: formData,
    });

    if (!extractResponse.ok) {
      throw new Error('Failed to analyze resume');
    }

    // Add a small delay to ensure file processing is complete
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Second API call to NLP module
    const nlpResponse = await fetch('http://localhost:8000/nlp-module', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!nlpResponse.ok) {
      const errorData = await nlpResponse.json();
      throw new Error(errorData.detail || 'Failed to process NLP analysis');
    }

    const nlpResults = await nlpResponse.json();
    setResults(nlpResults);
    console.log('NLP Results:', nlpResults);
    
  } catch (err) {
    setError(`Error: ${err.message}`);
    console.error('Error details:', err);
  } finally {
    setIsLoading(false);
  }
};

  const resetForm = () => {
    setFile(null);
    setRequirements('');
    setResults(null);
    setError('');
  };

  return (
    <div className="resume-parser">
      <header>
        <h1>Resume Parser</h1>
        <p>Check if your resume matches the job requirements</p>
      </header>

      {!results ? (
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="resume">Upload Resume (PDF)</label>
            <input 
              type="file" 
              id="resume" 
              accept=".pdf" 
              onChange={handleFileChange} 
            />
            {file && <div className="file-info">{file.name}</div>}
          </div>

          <div className="input-group">
            <label htmlFor="requirements">Job Requirements</label>
            <textarea
              id="requirements"
              placeholder="Enter keywords or requirements separated by commas (e.g., React, JavaScript, CSS, API integration)"
              value={requirements}
              onChange={handleRequirementsChange}
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button 
            type="submit" 
            className="analyze-button" 
            disabled={isLoading}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Resume'}
          </button>
        </form>
      ) : (
        <div className="results-container">
          <div className="results-header">
            <h2>Analysis Results</h2>
            <div className={`match-percentage ${results.suitable ? 'suitable' : 'not-suitable'}`}>
              {results.matchPercentage}% Match
            </div>
            <div className="suitability-indicator">
              {results.suitable 
                ? '✅ Your profile is suitable for this position' 
                : '❌ Your profile may not be suitable for this position'}
            </div>
          </div>

          <div className="keywords-results">
            <h3>Keywords Analysis</h3>
            
          </div>

          <button 
            className="reset-button" 
            onClick={resetForm}
          >
            Analyze Another Resume
          </button>
        </div>
      )}
    </div>
  );
}

export default ResumeParser;