from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import tempfile
import json
from typing import Dict, List
from pydantic import BaseModel, ConfigDict
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Add debug print to verify
print(f"DEBUG: GROQ_API_KEY loaded: {'Yes' if os.getenv('GROQ_API_KEY') else 'NO - CHECK .env FILE!'}")
if os.getenv('GROQ_API_KEY'):
    print(f"DEBUG: Key starts with: {os.getenv('GROQ_API_KEY')[:10]}...")

app = FastAPI(title="Meeting Analyzer API - 100% Free with Groq")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API client
groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not set. Get free API key from https://console.groq.com"
            )
        groq_client = Groq(api_key=api_key)
    return groq_client

# Pydantic v2 models
class MeetingAnalysis(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    transcript: str
    summary: str
    strengths: List[str]
    improvements: List[str]
    action_items: List[Dict[str, str]]
    key_decisions: List[str]
    participants: List[str]
    duration: str
    sentiment: str

@app.get("/")
async def root():
    return {
        "message": "Meeting Analyzer API - 100% Free with Groq",
        "transcription": "Groq Whisper API (free)",
        "analysis": "Groq Llama 3.3 70B (free)",
        "cost": "$0",
        "python_version": f"{os.sys.version}"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }

async def transcribe_audio(file_path: str, filename: str) -> str:
    """Transcribe audio using Groq Whisper API (free)"""
    try:
        client = get_groq_client()
        
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(filename, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                language="en"
            )
        
        return transcription.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def analyze_meeting(transcript: str) -> Dict:
    """Analyze meeting transcript using Groq AI (free)"""
    
    prompt = f"""As an expert meeting analyst and project management consultant, analyze this meeting transcript and provide comprehensive insights.

Transcript:
{transcript}

Provide a detailed JSON response with the following structure:
{{
  "summary": "A concise 2-3 sentence summary of the meeting",
  "strengths": ["List 4-5 specific strengths in the presentation/meeting delivery"],
  "improvements": ["List 4-5 specific, actionable improvements for better presentations"],
  "action_items": [
    {{"task": "specific task", "owner": "person/team responsible", "deadline": "when it's due"}},
    ...
  ],
  "key_decisions": ["List all important decisions made during the meeting"],
  "participants": ["List all participants mentioned"],
  "duration": "estimated duration in minutes",
  "sentiment": "overall meeting sentiment (positive/neutral/negative)"
}}

Focus on:
- Project management perspective
- Actionable feedback for presentation improvement
- Clear action items with ownership
- Professional communication insights

Respond ONLY with valid JSON, no additional text."""

    try:
        client = get_groq_client()
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert meeting analyst. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze")
async def analyze_recording(file: UploadFile = File(...)):
    """Upload and analyze a meeting recording - 100% FREE with Groq"""
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/ogg", "audio/webm", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an audio file (MP3, WAV, M4A, OGG, WebM, FLAC)"
        )
    
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file_path = temp_file.name
            
            content = await file.read()
            file_size = len(content)
            
            # Groq supports up to 25MB for audio files
            if file_size > 25 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="File too large. Groq API supports up to 25MB. Please use a shorter recording or compress the audio."
                )
            
            temp_file.write(content)
        
        print(f"Processing file: {file.filename} ({file_size / 1024 / 1024:.2f} MB)")
        
        print("Transcribing audio with Groq Whisper API...")
        transcript = await transcribe_audio(temp_file_path, file.filename)
        print(f"Transcription complete: {len(transcript)} characters")
        
        print("Analyzing with Groq Llama 3.3...")
        analysis = await analyze_meeting(transcript)
        
        result = {
            "transcript": transcript,
            **analysis
        }
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/api/generate-mom")
async def generate_minutes(analysis: MeetingAnalysis):
    """Generate formatted Minutes of Meeting document"""
    from datetime import datetime
    
    mom = f"""
MINUTES OF MEETING
Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
Duration: {analysis.duration}
Participants: {', '.join(analysis.participants)}

EXECUTIVE SUMMARY
{analysis.summary}

KEY DECISIONS
{chr(10).join([f"{i+1}. {decision}" for i, decision in enumerate(analysis.key_decisions)])}

ACTION ITEMS
{chr(10).join([f"{i+1}. {item['task']}\n   Owner: {item['owner']}\n   Deadline: {item['deadline']}" 
               for i, item in enumerate(analysis.action_items)])}

PRESENTATION STRENGTHS
{chr(10).join([f"{i+1}. {strength}" for i, strength in enumerate(analysis.strengths)])}

AREAS FOR IMPROVEMENT
{chr(10).join([f"{i+1}. {improvement}" for i, improvement in enumerate(analysis.improvements)])}

FULL TRANSCRIPT
{analysis.transcript}

---
Generated by Meeting Analyzer (100% Free with Groq)
Transcription: Groq Whisper-v3-turbo | Analysis: Llama 3.3 70B
"""
    
    return {"minutes": mom.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)