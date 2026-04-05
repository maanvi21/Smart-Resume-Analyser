import React, { useState } from 'react';
import './ResumeParser.css';

function ResumeParser() {
  const [file, setFile] = useState(null);
  const [requirements, setRequirements] = useState('');
  const [results, setResults] = useState(null);
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !requirements.trim()) {
      setError('Please upload a resume PDF and enter job requirements');
      return;
    }
    setIsLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('requirements', requirements);

      const extractResponse = await fetch('http://localhost:8000/extract-features', {
        method: 'POST',
        body: formData,
      });
      if (!extractResponse.ok) throw new Error('Failed to extract resume features');

      await new Promise(resolve => setTimeout(resolve, 1000));

      const nlpResponse = await fetch('http://localhost:8000/nlp-module', {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      if (!nlpResponse.ok) {
        const errorData = await nlpResponse.json();
        throw new Error(errorData.detail || 'Failed to process NLP analysis');
      }

      const nlpResults = await nlpResponse.json();
      setResults(nlpResults);
    } catch (err) {
      setError(`Error: ${err.message}`);
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
            <input type="file" id="resume" accept=".pdf" onChange={handleFileChange} />
            {file && <div className="file-info">{file.name}</div>}
          </div>

          <div className="input-group">
            <label htmlFor="requirements">Job Description</label>
            <textarea
              id="requirements"
              placeholder="Paste the full job description here — keywords will be extracted automatically"
              value={requirements}
              onChange={(e) => setRequirements(e.target.value)}
            />
          </div>

          {error && <div className="error-message">{error}</div>}

          <button type="submit" className="analyze-button" disabled={isLoading}>
            {isLoading ? 'Analyzing...' : 'Analyze Resume'}
          </button>
        </form>
      ) : (
        <div className="results-container">

          {/* ── Header ── */}
          <div className="results-header">
            <h2>Analysis Results</h2>
            <div className={`match-percentage ${results.suitable ? 'suitable' : 'not-suitable'}`}>
              {results.semantic_match_percentage}% Match
            </div>
            <div className="suitability-indicator">
              {results.suitable
                ? '✅ Your profile is suitable for this position'
                : '❌ Your profile may not be suitable for this position'}
            </div>
          </div>

          {/* ── Semantic scores per requirement ── */}
          <div className="section">
            <h3>Requirement Scores</h3>
            <div className="score-list">
              {Object.entries(results.per_requirement_scores || {})
                .sort((a, b) => b[1] - a[1])
                .map(([req, score]) => (
                  <div key={req} className="score-row">
                    <span className="score-label">{req}</span>
                    <div className="score-bar-bg">
                      <div
                        className="score-bar-fill"
                        style={{
                          width: `${Math.round(score * 100)}%`,
                          background: score >= 0.65 ? '#3B6D11' : score >= 0.40 ? '#BA7517' : '#A32D2D'
                        }}
                      />
                    </div>
                    <span className="score-pct">{Math.round(score * 100)}%</span>
                  </div>
                ))}
            </div>
          </div>

          {/* ── Matched / Unmatched ── */}
          <div className="section">
            <h3>Keywords Analysis</h3>
            <div className="keywords-grid">
              <div>
                <p className="kw-heading matched-heading">✅ Matched</p>
                <div className="chips">
                  {(results.matched_requirements || []).map(k => (
                    <span key={k} className="chip chip-match">{k}</span>
                  ))}
                  {(results.matched_requirements || []).length === 0 && (
                    <span className="no-data">None</span>
                  )}
                </div>
              </div>
              <div>
                <p className="kw-heading unmatched-heading">❌ Missing</p>
                <div className="chips">
                  {(results.unmatched_requirements || []).map(k => (
                    <span key={k} className="chip chip-miss">{k}</span>
                  ))}
                  {(results.unmatched_requirements || []).length === 0 && (
                    <span className="no-data">None — great match!</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ── Named Entities ── */}
          {results.entities && Object.keys(results.entities).length > 0 && (
            <div className="section">
              <h3>Entities Found</h3>
              <div className="entities-grid">
                {Object.entries(results.entities).map(([group, items]) => (
                  <div key={group} className="entity-group">
                    <p className="entity-group-label">{group}</p>
                    <div className="chips">
                      {items.map(item => (
                        <span key={item} className={`chip chip-${group.toLowerCase()}`}>{item}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Bullet Rewrites ── */}
          {results.rewritten_bullets && results.rewritten_bullets.length > 0 && (
            <div className="section">
              <h3>AI Bullet Rewrites</h3>
              {results.rewritten_bullets.map((item, i) => (
                <div key={i} className="bullet-card">
                  <div className="bullet-before">
                    <span className="bullet-tag before-tag">Before</span>
                    {item.original}
                  </div>
                  <div className="bullet-after">
                    <span className="bullet-tag after-tag">After</span>
                    {item.improved}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* ── Missing Skill Suggestions ── */}
          {results.missing_skill_suggestions && results.missing_skill_suggestions.length > 0 && (
            <div className="section">
              <h3>Suggestions for Missing Skills</h3>
              {results.missing_skill_suggestions.map((s, i) => (
                <div key={i} className="suggestion-card">
                  <span className="suggestion-skill">{typeof s === "object" ? s.skill : s}</span>
                  <p className="suggestion-text">{typeof s === "object" ? s.suggestion : ""}</p>
                </div>
              ))}
            </div>
          )}

          <button className="reset-button" onClick={resetForm}>
            Analyze Another Resume
          </button>
        </div>
      )}
    </div>
  );
}

export default ResumeParser;