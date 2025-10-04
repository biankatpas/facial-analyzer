"""
FastAPI - Facial Expression Analyzer API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import os
from typing import List, Dict, Optional
from datetime import datetime
from src.analyzer import FacialExpressionAnalyzer

app = FastAPI(
    title="Facial Analyzer API",
    description="Real-time facial expression analyzer for HR",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar analyzer
gemini_key = os.getenv('GEMINI_API_KEY')
analyzer = FacialExpressionAnalyzer(gemini_api_key=gemini_key)

# Memory session storage
sessions = {}


# Models
class EmotionResponse(BaseModel):
    emotions: Dict[str, float]
    dominant: str
    timestamp: str


class AnalysisSession(BaseModel):
    session_id: str
    question: str
    started_at: str


class SummaryResponse(BaseModel):
    emotion: str
    media: float
    max: float
    min: float


# Routes
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gemini_configured": analyzer.model is not None
    }


@app.post("/session/start")
def start_session(question: str):
    """Inicia uma nova sessão de análise"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    sessions[session_id] = {
        "question": question,
        "started_at": datetime.now().isoformat(),
        "emotions": []
    }
    
    return {
        "session_id": session_id,
        "question": question,
        "started_at": sessions[session_id]["started_at"]
    }


@app.post("/analyze/frame", response_model=EmotionResponse)
async def analyze_frame(file: UploadFile = File(...), session_id: Optional[str] = None):
    """
    Analisa um frame de vídeo e retorna as emoções detectadas
    """
    try:
        # Ler imagem do upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Analisar emoção
        emotion_data = analyzer.analyze_emotion(frame)
        
        if not emotion_data:
            raise HTTPException(status_code=500, detail="Could not analyze emotions")
        
        # Armazenar na sessão se fornecido session_id
        if session_id and session_id in sessions:
            sessions[session_id]["emotions"].append(emotion_data)
        
        return emotion_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/faces")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detecta faces em um frame
    """
    try:
        # Ler imagem
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detectar faces
        faces = analyzer.detect_face(frame)
        
        return {
            "faces_detected": len(faces),
            "faces": [
                {"x": x, "y": y, "width": w, "height": h}
                for x, y, w, h in faces
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/summary")
def get_session_summary(session_id: str):
    """
    Retorna o resumo de uma sessão de análise
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if not session["emotions"]:
        return {
            "session_id": session_id,
            "question": session["question"],
            "total_frames": 0,
            "summary": {}
        }
    
    # Gerar resumo
    summary = analyzer.generate_emotion_summary(session["emotions"])
    
    return {
        "session_id": session_id,
        "question": session["question"],
        "started_at": session["started_at"],
        "total_frames": len(session["emotions"]),
        "summary": summary
    }


@app.post("/session/{session_id}/insights")
def generate_insights(session_id: str):
    """
    Gera insights para uma sessão
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if not session["emotions"]:
        raise HTTPException(status_code=400, detail="No emotions data to analyze")
    
    # Gerar resumo e insights
    summary = analyzer.generate_emotion_summary(session["emotions"])
    insights = analyzer.generate_ai_insights(session["question"], summary)
    
    return {
        "session_id": session_id,
        "question": session["question"],
        "summary": summary,
        "insights": insights
    }


@app.get("/sessions")
def list_sessions():
    """Lista todas as sessões ativas"""
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "question": data["question"],
                "started_at": data["started_at"],
                "frames_analyzed": len(data["emotions"])
            }
            for sid, data in sessions.items()
        ]
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Remove uma sessão"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}


@app.post("/export/report")
def export_report():
    """
    Exporta relatório completo de todas as sessões
    """
    try:
        # Preparar dados para export
        for session_id, session_data in sessions.items():
            if session_data["emotions"]:
                analyzer.analysis_results.append({
                    'question': session_data["question"],
                    'emotions': session_data["emotions"],
                    'duration': 0,  # Calcular se necessário
                    'timestamp': session_data["started_at"]
                })
        
        # Exportar relatório
        report = analyzer.export_report()
        
        return {
            "message": "Report exported successfully",
            "report": report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
