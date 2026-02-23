from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import io
from typing import Optional
from fastapi.encoders import jsonable_encoder
import asyncio
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stress_detector import StressDetector
from models.recommendation_engine import RecommendationEngine
from models.health_risk_model import predict_health_risk
from models.symptom_analyzer import SymptomAnalyzer
from config import Config
import requests
import numpy as np

app = FastAPI(
    title="Stress Monitoring System API",
    description="AI/ML-powered stress monitoring with personalized recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
stress_detector = None
recommendation_engine = None
symptom_analyzer = SymptomAnalyzer()

# Simple in-memory connection manager for WebSocket broadcasting
class ConnectionManager:
    def __init__(self):
        self._active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._active.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self._active:
                self._active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        data = jsonable_encoder(_sanitize_numpy(message))
        async with self._lock:
            websockets = list(self._active)
        for ws in websockets:
            try:
                await ws.send_json(data)
            except Exception:
                # Drop dead connections silently
                await self.disconnect(ws)

manager = ConnectionManager()

# Pydantic models for API
class StressData(BaseModel):
    heart_rate: float
    sleep_hours: float
    activity_level: float
    mood_score: float
    work_stress: float
    social_interaction: float
    caffeine_intake: float
    exercise_frequency: float
    age: int
    gender: int  # 0: Female, 1: Male

class UserPreferences(BaseModel):
    interests: Optional[List[str]] = []
    available_time: Optional[str] = "1-2 hours daily"
    preferred_activities: Optional[List[str]] = []
    music_preferences: Optional[List[str]] = []
    book_genres: Optional[List[str]] = []

class StressPrediction(BaseModel):
    stress_level: int
    stress_label: str
    confidence: float
    probabilities: Dict[str, float]

class StreamSample(BaseModel):
    heart_rate: float
    sleep_hours: float
    activity_level: float
    mood_score: float
    work_stress: float
    social_interaction: float
    caffeine_intake: float
    exercise_frequency: float
    age: int
    gender: int
    timestamp: Optional[str] = None  # ISO8601 (client-supplied optional)

class Recommendations(BaseModel):
    immediate_actions: List[str]
    books: List[str]
    music: List[str]
    activities: List[str]
    meditation: List[str]
    social: List[str]
    long_term: List[str]

class CoachChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class CoachChatResponse(BaseModel):
    reply: str
    sos: bool = False
    disclaimer: str = "This assistant provides general wellness information and is not a substitute for professional medical advice. If you are in crisis, seek immediate help."

class SymptomChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class SymptomChatResponse(BaseModel):
    role: str
    content: str
    question: Optional[str] = None
    needs_followup: bool
    symptom: Optional[str] = None
    analysis: Optional[Dict] = None

class AppointmentRequest(BaseModel):
    doctor_id: str
    patient_name: str
    patient_email: str
    patient_phone: str
    preferred_date: str
    preferred_time: str
    reason: str


class HealthData(BaseModel):
    age: float
    gender: str
    systolic_bp: float
    diastolic_bp: float
    blood_sugar: float
    bmi: float
    cholesterol: float
    smoking: str
    activity_level: str


class HealthPrediction(BaseModel):
    risk: str
    explanation: str
    recommendations: List[str]

# Dependency to get stress detector
def get_stress_detector():
    global stress_detector
    if stress_detector is None:
        stress_detector = StressDetector("random_forest")
        # Try to load existing model, otherwise train a new one
        try:
            stress_detector.load_model(Config.MODEL_PATH, Config.SCALER_PATH)
        except:
            print("No existing model found, training new model...")
            stress_detector.train()
            stress_detector.save_model(Config.MODEL_PATH, Config.SCALER_PATH)
    return stress_detector

# Dependency to get recommendation engine
def get_recommendation_engine():
    global recommendation_engine
    if recommendation_engine is None:
        try:
            recommendation_engine = RecommendationEngine()
        except ValueError as e:
            print(f"Warning: {e}")
            recommendation_engine = None
    return recommendation_engine

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("Initializing stress monitoring system...")
    # Initialize stress detector
    get_stress_detector()
    # Initialize recommendation engine
    get_recommendation_engine()
    print("System initialized successfully!")

def _sanitize_numpy(obj: Any) -> Any:
    """Recursively convert NumPy types to plain Python for JSON encoding."""
    try:
        if isinstance(obj, dict):
            return {k: _sanitize_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_numpy(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_sanitize_numpy(x) for x in obj)
        # Scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    except Exception:
        return obj

def _triage_needs_sos(text: str) -> bool:
    if not text:
        return False
    keywords = [
        "suicide", "kill myself", "self harm", "overdose", "can't breathe",
        "chest pain", "heart attack", "stroke", "fainting", "unconscious"
    ]
    lowered = text.lower()
    return any(k in lowered for k in keywords)

@app.post("/api/coach/chat", response_model=CoachChatResponse)
async def coach_chat(req: CoachChatRequest):
    """Proxy chat to local Ollama with minimal safety triage."""
    try:
        sos = _triage_needs_sos(req.message)
        if sos:
            reply = (
                "This sounds urgent. Please seek immediate help: call local emergency services or visit the nearest ER. "
                "Try slow breathing while you get help: inhale 4s, hold 4s, exhale 6s."
            )
            return CoachChatResponse(reply=reply, sos=True)

        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{ollama_host.rstrip('/')}/api/generate"
        system_prompt = (
            "You are a supportive wellness assistant. Provide calm, practical, evidence-informed guidance for stress relief. "
            "Do not diagnose. Suggest simple actions (breathing, grounding, hydration, short walk, social support). "
            "Answer concisely: use at most 3 short bullet points and a single-line caution when urgent care is needed."
        )
        # Build a concise prompt with optional short history
        context = "\n".join(
            [f"{m.get('role','user')}: {m.get('content','')}" for m in (req.history or [])][-6:]
        )
        prompt = f"System: {system_prompt}\n{context}\nUser: {req.message}\nAssistant:"
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3"),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        r = requests.post(url, json=payload, timeout=60)
        if not r.ok:
            raise HTTPException(status_code=502, detail="Ollama not reachable or returned error. Ensure it is running and OLLAMA_HOST/OLLAMA_MODEL are set.")
        data = r.json()
        # Be tolerant to different response shapes
        reply = (
            data.get("response")
            or data.get("message")
            or (data.get("choices", [{}])[0] or {}).get("message", {}).get("content")
            or ""
        )
        if isinstance(reply, dict):
            # Some backends may nest content
            reply = reply.get("content", "")
        if not reply:
            raise HTTPException(status_code=502, detail="Empty response from Ollama.")
        return CoachChatResponse(reply=reply, sos=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coach chat failed: {str(e)}")

@app.post("/api/stream/ingest")
async def ingest_stream_sample(sample: StreamSample, detector: StressDetector = Depends(get_stress_detector)):
    """Ingest a single wearable/phone sample, run prediction, and broadcast to subscribers."""
    try:
        payload = sample.dict()
        # Ensure timestamp
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Build features expected by the model
        features = {k: payload[k] for k in [
            "heart_rate","sleep_hours","activity_level","mood_score","work_stress",
            "social_interaction","caffeine_intake","exercise_frequency","age","gender"
        ]}

        pred = _sanitize_numpy(detector.predict(features))
        event = _sanitize_numpy({
            "type": "prediction",
            "timestamp": payload["timestamp"],
            "input": features,
            "output": pred,
        })
        # Broadcast asynchronously (fire-and-forget)
        asyncio.create_task(manager.broadcast(event))
        return event
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep the connection alive; server pushes predictions on ingest
        while True:
            # Optionally receive pings or client filters; for now, just await to detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve unified home page with navigation to Health and Stress modules."""
    try:
        with open("frontend/home.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to stress dashboard if home not found
        return await stress_page()


@app.get("/stress", response_class=HTMLResponse)
async def stress_page():
    """Serve the advanced stress dashboard."""
    try:
        with open("frontend/advanced_dashboard.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open("frontend/index.html", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return HTMLResponse(
                "<h1>Stress module UI not found</h1>",
                status_code=500,
            )


@app.get("/health", response_class=HTMLResponse)
async def health_page():
    """Serve the health checkup UI."""
    try:
        with open("frontend/health_check.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>Health checkup UI not found</h1>",
            status_code=500,
        )

@app.post("/api/stress/predict", response_model=StressPrediction)
async def predict_stress(
    stress_data: StressData,
    detector: StressDetector = Depends(get_stress_detector)
):
    """Predict stress level based on user data"""
    try:
        features = stress_data.dict()
        prediction = detector.predict(features)
        return StressPrediction(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/recommendations", response_model=Recommendations)
async def get_recommendations(
    stress_data: StressData,
    user_preferences: UserPreferences = None,
    detector: StressDetector = Depends(get_stress_detector),
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized recommendations based on stress level"""
    try:
        # First predict stress level
        features = stress_data.dict()
        stress_prediction = detector.predict(features)
        
        # Generate recommendations
        if engine:
            recommendations = engine.generate_recommendations(
                stress_prediction, 
                user_preferences.dict() if user_preferences else None
            )
        else:
            # Fallback to basic recommendations
            recommendations = {
                'immediate_actions': ["Take deep breaths", "Step away from stressor"],
                'books': ["The Power of Now", "Atomic Habits"],
                'music': ["Calming instrumental music", "Nature sounds"],
                'activities': ["Gentle exercise", "Meditation"],
                'meditation': ["10-minute breathing exercise", "Body scan"],
                'social': ["Talk to a friend", "Join support group"],
                'long_term': ["Regular exercise", "Stress management training"]
            }
        
        return Recommendations(**recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@app.get("/api/emergency")
async def get_emergency_recommendations(
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get immediate emergency stress relief recommendations"""
    try:
        if engine:
            recommendations = engine.get_emergency_recommendations()
        else:
            recommendations = [
                "Stop and take 5 deep breaths",
                "Use the 5-4-3-2-1 grounding technique",
                "Step away from the current situation",
                "Call a trusted friend or family member"
            ]
        return {"emergency_recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get emergency recommendations: {str(e)}")

@app.get("/api/weekly-plan")
async def get_weekly_plan(
    stress_level: int = 0,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get a weekly stress management plan"""
    try:
        if engine:
            plan = engine.get_weekly_plan(stress_level)
        else:
            plan = {
                'monday': ["Morning meditation", "Regular exercise"],
                'tuesday': ["Work-life balance check", "Social activity"],
                'wednesday': ["Mid-week stress check", "Nature walk"],
                'thursday': ["Professional development", "Social connection"],
                'friday': ["Week reflection", "Fun activity"],
                'saturday': ["Family time", "Outdoor activity"],
                'sunday': ["Rest and recovery", "Next week planning"]
            }
        return {"weekly_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get weekly plan: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check system health and model status"""
    try:
        detector_status = "Ready" if stress_detector and stress_detector.is_trained else "Not Ready"
        engine_status = "Ready" if recommendation_engine else "Not Ready (API key required)"
        
        return {
            "status": "Healthy",
            "models": {
                "stress_detector": detector_status,
                "recommendation_engine": engine_status
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "Unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }


@app.post("/api/health/predict", response_model=HealthPrediction)
async def predict_health(data: HealthData):
    """
    Predict overall health risk using the AI Hackathon model.
    """
    try:
        risk = predict_health_risk(
            age=data.age,
            gender=data.gender,
            systolic_bp=data.systolic_bp,
            diastolic_bp=data.diastolic_bp,
            blood_sugar=data.blood_sugar,
            bmi=data.bmi,
            cholesterol=data.cholesterol,
            smoking=data.smoking,
            activity_level=data.activity_level,
        )

        explanation_map = {
            "Low": "Your metrics are mostly within healthy ranges.",
            "Medium": "Some metrics are elevated. Monitoring and lifestyle tweaks are advised.",
            "High": "Multiple risk factors detected. Consult a healthcare professional.",
        }
        recommendations_map = {
            "Low": [
                "Maintain balanced diet and regular activity.",
                "Keep routine health check-ups.",
                "Stay hydrated and sleep 7-8 hours.",
            ],
            "Medium": [
                "Increase physical activity to at least 150 minutes/week.",
                "Reduce sugar and saturated fats; prioritize vegetables.",
                "Monitor blood pressure and glucose regularly.",
            ],
            "High": [
                "Consult a doctor for personalized guidance.",
                "Adopt a heart-healthy diet (DASH/Mediterranean).",
                "Avoid smoking and limit alcohol.",
            ],
        }

        explanation = explanation_map.get(risk, "Risk level unavailable.")
        recs = recommendations_map.get(risk, [])
        return HealthPrediction(
            risk=risk,
            explanation=explanation,
            recommendations=recs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health prediction failed: {str(e)}")

@app.get("/api/sample-data")
async def get_sample_data():
    """Get sample stress data for testing"""
    return {
        "sample_stress_data": {
            "heart_rate": 85.0,
            "sleep_hours": 6.5,
            "activity_level": 0.4,
            "mood_score": 6.0,
            "work_stress": 7.0,
            "social_interaction": 5.0,
            "caffeine_intake": 2.0,
            "exercise_frequency": 3.0,
            "age": 30,
            "gender": 1
        },
        "sample_preferences": {
            "interests": ["reading", "music", "nature"],
            "available_time": "1-2 hours daily",
            "preferred_activities": ["walking", "meditation", "reading"],
            "music_preferences": ["classical", "ambient", "jazz"],
            "book_genres": ["self-help", "psychology", "fiction"]
        }
    }

@app.get("/api/dataset")
async def get_dataset(n: int = 1000):
    """Return a generated sample dataset (JSON) with metadata"""
    try:
        detector = get_stress_detector()
        import pandas as pd  # local import to avoid startup cost if unused
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)
        preview = df.head(50).to_dict(orient="records")
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "preview": preview
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build dataset: {str(e)}")

@app.get("/download/dataset.csv")
async def download_dataset_csv(n: int = 1000):
    """Download a generated sample dataset as CSV"""
    try:
        detector = get_stress_detector()
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stress_dataset_{n}.csv"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

@app.get("/api/dataset/filter")
async def filter_dataset(
    n: int = 1000,
    stress_level: Optional[int] = None,
    min_sleep: Optional[float] = None,
    max_sleep: Optional[float] = None,
    min_hr: Optional[float] = None,
    max_hr: Optional[float] = None,
    min_work_stress: Optional[float] = None,
    max_work_stress: Optional[float] = None,
):
    """Filter the generated dataset and return a compact summary + preview"""
    try:
        detector = get_stress_detector()
        import pandas as pd
        n = max(10, min(n, 10000))
        df = detector.create_sample_data(n)

        # Apply filters
        if stress_level is not None:
            df = df[df["stress_level"] == int(stress_level)]
        if min_sleep is not None:
            df = df[df["sleep_hours"] >= float(min_sleep)]
        if max_sleep is not None:
            df = df[df["sleep_hours"] <= float(max_sleep)]
        if min_hr is not None:
            df = df[df["heart_rate"] >= float(min_hr)]
        if max_hr is not None:
            df = df[df["heart_rate"] <= float(max_hr)]
        if min_work_stress is not None:
            df = df[df["work_stress"] >= float(min_work_stress)]
        if max_work_stress is not None:
            df = df[df["work_stress"] <= float(max_work_stress)]

        # Build summary
        summary = df.describe(include="all").to_dict()
        preview = df.head(50).to_dict(orient="records")
        return {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "summary": summary,
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to filter dataset: {str(e)}")

@app.get("/api/analytics/trends")
async def get_analytics_trends():
    """Get stress trend analytics"""
    return {
        "weekly_trend": [
            {"day": "Monday", "stress_level": 2, "mood": 6, "activity": 0.7},
            {"day": "Tuesday", "stress_level": 3, "mood": 5, "activity": 0.6},
            {"day": "Wednesday", "stress_level": 1, "mood": 8, "activity": 0.8},
            {"day": "Thursday", "stress_level": 4, "mood": 4, "activity": 0.5},
            {"day": "Friday", "stress_level": 2, "mood": 7, "activity": 0.6},
            {"day": "Saturday", "stress_level": 1, "mood": 9, "activity": 0.9},
            {"day": "Sunday", "stress_level": 1, "mood": 8, "activity": 0.7}
        ],
        "monthly_insights": {
            "avg_stress": 2.1,
            "improvement": 15.3,
            "best_day": "Saturday",
            "challenging_day": "Thursday"
        },
        "correlations": {
            "sleep_vs_stress": -0.78,
            "activity_vs_mood": 0.65,
            "work_stress_vs_overall": 0.82
        }
    }

@app.get("/api/ai-coach/insights")
async def get_ai_coach_insights():
    """Get AI coach personalized insights"""
    return {
        "personalized_message": "Based on your recent patterns, I notice you tend to feel more stressed on Thursdays. Let's work on some Thursday-specific strategies!",
        "recommendations": [
            "Try a 5-minute breathing exercise every Thursday morning",
            "Schedule lighter tasks for Thursday afternoons",
            "Plan a relaxing activity for Thursday evenings"
        ],
        "progress_tips": [
            "Your sleep quality has improved 20% this week!",
            "Consider adding 10 more minutes of daily activity",
            "Your stress management techniques are working well"
        ],
        "next_goals": [
            "Maintain your current sleep schedule",
            "Try one new relaxation technique this week",
            "Connect with a friend or family member"
        ]
    }

@app.get("/api/gamification/achievements")
async def get_achievements():
    """Get user achievements and progress"""
    return {
        "achievements": [
            {
                "id": "first_check",
                "title": "First Steps",
                "description": "Completed your first stress assessment",
                "icon": "🎯",
                "unlocked": True,
                "date": "2024-01-15"
            },
            {
                "id": "week_streak",
                "title": "Consistency Champion",
                "description": "Used the app for 7 consecutive days",
                "icon": "🔥",
                "unlocked": True,
                "date": "2024-01-22"
            },
            {
                "id": "stress_master",
                "title": "Stress Master",
                "description": "Maintained low stress for 5 days straight",
                "icon": "🧘",
                "unlocked": False,
                "progress": 3
            },
            {
                "id": "community_helper",
                "title": "Community Helper",
                "description": "Helped 5 community members",
                "icon": "🤝",
                "unlocked": False,
                "progress": 2
            }
        ],
        "current_streak": 12,
        "total_points": 1250,
        "level": "Wellness Warrior",
        "next_level_points": 500
    }

@app.get("/api/wearable/sync")
async def sync_wearable_data():
    """Simulate wearable device data sync"""
    import random
    return {
        "heart_rate": {
            "current": random.randint(60, 100),
            "avg_resting": random.randint(55, 75),
            "variability": random.uniform(20, 50)
        },
        "sleep": {
            "duration": random.uniform(6, 9),
            "quality": random.uniform(70, 95),
            "deep_sleep": random.uniform(15, 25),
            "rem_sleep": random.uniform(20, 30)
        },
        "activity": {
            "steps": random.randint(5000, 15000),
            "calories": random.randint(1800, 2500),
            "active_minutes": random.randint(30, 120)
        },
        "stress_indicators": {
            "hrv": random.uniform(30, 60),
            "skin_temp": random.uniform(36, 37),
            "movement_restlessness": random.uniform(0, 1)
        }
    }

@app.post("/api/community/share")
async def share_achievement(achievement_data: dict):
    """Share achievement with community"""
    return {
        "success": True,
        "message": "Achievement shared successfully!",
        "community_response": {
            "likes": 15,
            "comments": 3,
            "encouragement": "Great job! Keep up the amazing work! 💪"
        }
    }

@app.get("/api/meditation/sessions")
async def get_meditation_sessions():
    """Get available meditation sessions"""
    return {
        "sessions": [
            {
                "id": "breathing_basics",
                "title": "5-Minute Breathing Basics",
                "duration": 5,
                "difficulty": "Beginner",
                "description": "Perfect for beginners to start their meditation journey",
                "audio_url": "/api/meditation/audio/breathing_basics"
            },
            {
                "id": "stress_relief",
                "title": "10-Minute Stress Relief",
                "duration": 10,
                "difficulty": "Intermediate",
                "description": "Designed specifically for stress reduction",
                "audio_url": "/api/meditation/audio/stress_relief"
            },
            {
                "id": "body_scan",
                "title": "15-Minute Body Scan",
                "duration": 15,
                "difficulty": "Intermediate",
                "description": "Progressive relaxation technique for deep calm",
                "audio_url": "/api/meditation/audio/body_scan"
            }
        ]
    }

@app.get("/api/ai-coach/voice-interaction")
async def get_voice_interaction():
    """Get AI coach voice interaction capabilities"""
    return {
        "voice_enabled": True,
        "available_commands": [
            "Tell me about my stress level",
            "Give me a breathing exercise",
            "What should I do right now?",
            "Play calming music",
            "Schedule a meditation session"
        ],
        "current_mood": "calm",
        "suggested_response": "I sense you might be feeling a bit overwhelmed. Would you like to try a quick breathing exercise together?"
    }

# Symptom Analysis and AI Doctor Endpoints
@app.post("/api/precautions/chat", response_model=SymptomChatResponse)
async def symptom_chat(request: SymptomChatRequest):
    """AI Doctor chat interface with real-time AI for medical questions"""
    try:
        conversation_history = request.conversation_history or []
        
        # Check for emergency keywords
        sos = _triage_needs_sos(request.message)
        if sos:
            return SymptomChatResponse(
                role="assistant",
                content="This sounds urgent. Please seek immediate help: call local emergency services or visit the nearest ER. Try slow breathing while you get help: inhale 4s, hold 4s, exhale 6s.",
                question=None,
                needs_followup=False,
                symptom=None,
                analysis=None
            )
        
        # Use real AI (Ollama) for medical conversations
        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{ollama_host.rstrip('/')}/api/generate"
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "\n".join([
                f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
                for msg in conversation_history[-6:]  # Last 6 messages for context
            ])
        
        # Enhanced system prompt for medical AI
        system_prompt = (
            "You are a knowledgeable medical AI assistant. Your role is to:\n"
            "1. Help users understand their symptoms and health concerns\n"
            "2. Ask relevant follow-up questions to better understand their condition\n"
            "3. Provide evidence-based dos and don'ts for their specific situation\n"
            "4. Explain possible causes in simple terms\n"
            "5. Know when to recommend seeing a doctor\n\n"
            "Guidelines:\n"
            "- Be empathetic and supportive\n"
            "- Ask 1-2 specific follow-up questions when needed (e.g., 'What did you eat?', 'How long have you had this?', 'Any other symptoms?')\n"
            "- Provide clear, actionable dos and don'ts\n"
            "- Always remind users this is not a substitute for professional medical advice\n"
            "- If symptoms are severe or persistent, recommend seeing a healthcare professional\n"
            "- Keep responses concise but informative (2-4 sentences + questions/recommendations)\n"
            "- For casual greetings, introduce yourself and ask how you can help\n"
        )
        
        # Build the prompt
        if context:
            prompt = f"{system_prompt}\n\nPrevious conversation:\n{context}\n\nUser: {request.message}\nAssistant:"
        else:
            prompt = f"{system_prompt}\n\nUser: {request.message}\nAssistant:"
        
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "llama3"),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Slightly higher for more natural conversation
                "top_p": 0.9
            }
        }
        
        try:
            r = requests.post(url, json=payload, timeout=60)
            if not r.ok:
                raise Exception("Ollama not reachable")
            
            data = r.json()
            reply = (
                data.get("response") or
                data.get("message") or
                (data.get("choices", [{}])[0] or {}).get("message", {}).get("content") or
                ""
            )
            
            if isinstance(reply, dict):
                reply = reply.get("content", "")
            
            if not reply:
                raise Exception("Empty response from Ollama")
            
            # Detect symptom from conversation for doctor recommendations
            detected_symptom = symptom_analyzer.detect_symptom(request.message)
            if not detected_symptom and conversation_history:
                # Try to detect from conversation history
                for msg in conversation_history:
                    detected = symptom_analyzer.detect_symptom(msg.get("content", ""))
                    if detected:
                        detected_symptom = detected
                        break
            
            # Determine if we need follow-up or can provide analysis
            needs_followup = True
            analysis = None
            
            # If we have enough context and a detected symptom, try to provide analysis
            if detected_symptom and len(conversation_history) >= 2:
                # Check if user seems to have provided enough info
                message_lower = request.message.lower()
                if any(keyword in message_lower for keyword in ["eat", "drank", "hours", "days", "pain", "feel", "symptom"]):
                    # Generate analysis
                    analysis = symptom_analyzer.analyze_symptoms(detected_symptom, conversation_history)
                    needs_followup = False
                    # Enhance AI reply with structured recommendations
                    reply += f"\n\nBased on our conversation, here are some specific recommendations for {detected_symptom}:"
            
            return SymptomChatResponse(
                role="assistant",
                content=reply,
                question=None,
                needs_followup=needs_followup,
                symptom=detected_symptom,
                analysis=analysis
            )
            
        except Exception as ai_error:
            # If AI is unavailable, do NOT fall back to fixed questions.
            # Instead, return a clear instruction to start Ollama so the user gets "real AI".
            print(f"AI error in /api/precautions/chat: {ai_error}")
            return SymptomChatResponse(
                role="assistant",
                content=(
                    "Real-time AI is currently offline because Ollama is not reachable.\n\n"
                    "To enable it:\n"
                    "1) Install Ollama and start it (or run `ollama serve`).\n"
                    "2) Pull a model: `ollama pull llama3` (or set OLLAMA_MODEL).\n"
                    "3) Ensure it is running at `http://127.0.0.1:11434` (or set OLLAMA_HOST).\n\n"
                    "After that, refresh this page and try again."
                ),
                question=None,
                needs_followup=False,
                symptom=None,
                analysis=None,
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symptom chat failed: {str(e)}")

@app.get("/api/precautions/doctors")
async def get_doctors(specialty: Optional[str] = None, symptom: Optional[str] = None):
    """Get list of available doctors, optionally filtered by specialty or symptom"""
    # Sample doctor database
    all_doctors = [
        {
            "id": "doc_001",
            "name": "Dr. Sarah Johnson",
            "specialty": "General Medicine",
            "qualifications": "MD, MBBS",
            "experience": "15 years",
            "rating": 4.8,
            "available_slots": ["09:00", "10:00", "11:00", "14:00", "15:00"],
            "location": "City Hospital, Downtown",
            "phone": "+1-234-567-8901",
            "email": "sarah.johnson@hospital.com"
        },
        {
            "id": "doc_002",
            "name": "Dr. Michael Chen",
            "specialty": "Gastroenterology",
            "qualifications": "MD, Gastroenterology",
            "experience": "12 years",
            "rating": 4.9,
            "available_slots": ["09:30", "10:30", "13:00", "14:30", "16:00"],
            "location": "Medical Center, Uptown",
            "phone": "+1-234-567-8902",
            "email": "michael.chen@hospital.com"
        },
        {
            "id": "doc_003",
            "name": "Dr. Emily Rodriguez",
            "specialty": "General Medicine",
            "qualifications": "MD, Internal Medicine",
            "experience": "10 years",
            "rating": 4.7,
            "available_slots": ["08:00", "09:00", "10:00", "13:00", "14:00", "15:00"],
            "location": "Community Clinic, Midtown",
            "phone": "+1-234-567-8903",
            "email": "emily.rodriguez@hospital.com"
        },
        {
            "id": "doc_004",
            "name": "Dr. James Wilson",
            "specialty": "Neurology",
            "qualifications": "MD, Neurology",
            "experience": "18 years",
            "rating": 4.9,
            "available_slots": ["10:00", "11:00", "14:00", "15:00", "16:00"],
            "location": "Neurology Center, Eastside",
            "phone": "+1-234-567-8904",
            "email": "james.wilson@hospital.com"
        },
        {
            "id": "doc_005",
            "name": "Dr. Priya Patel",
            "specialty": "Pulmonology",
            "qualifications": "MD, Pulmonology",
            "experience": "14 years",
            "rating": 4.8,
            "available_slots": ["09:00", "10:30", "11:30", "14:00", "15:30"],
            "location": "Respiratory Care Center",
            "phone": "+1-234-567-8905",
            "email": "priya.patel@hospital.com"
        },
        {
            "id": "doc_006",
            "name": "Dr. Robert Kim",
            "specialty": "General Medicine",
            "qualifications": "MD, Family Medicine",
            "experience": "20 years",
            "rating": 4.9,
            "available_slots": ["08:30", "09:30", "10:30", "13:30", "14:30"],
            "location": "Family Health Center",
            "phone": "+1-234-567-8906",
            "email": "robert.kim@hospital.com"
        },
        {
            "id": "doc_007",
            "name": "Dr. Lisa Anderson",
            "specialty": "ENT",
            "qualifications": "MD, ENT Specialist",
            "experience": "11 years",
            "rating": 4.7,
            "available_slots": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
            "location": "ENT Clinic, Westside",
            "phone": "+1-234-567-8907",
            "email": "lisa.anderson@hospital.com"
        },
        {
            "id": "doc_008",
            "name": "Dr. David Thompson",
            "specialty": "Gastroenterology",
            "qualifications": "MD, Gastroenterology",
            "experience": "16 years",
            "rating": 4.8,
            "available_slots": ["08:00", "09:00", "13:00", "14:00", "15:00"],
            "location": "Digestive Health Institute",
            "phone": "+1-234-567-8908",
            "email": "david.thompson@hospital.com"
        }
    ]
    
    # Filter by specialty if provided
    if specialty:
        filtered = [d for d in all_doctors if specialty.lower() in d["specialty"].lower()]
        return {"doctors": filtered}
    
    # Filter by symptom if provided
    if symptom:
        symptom_lower = symptom.lower()
        symptom_specialties = {
            "fever": ["General Medicine", "Internal Medicine"],
            "stomach": ["Gastroenterology", "General Medicine"],
            "headache": ["Neurology", "General Medicine"],
            "cough": ["Pulmonology", "General Medicine"],
            "cold": ["General Medicine", "ENT"]
        }
        
        specialties = symptom_specialties.get(symptom_lower, ["General Medicine"])
        filtered = [d for d in all_doctors if any(spec.lower() in d["specialty"].lower() for spec in specialties)]
        return {"doctors": filtered}
    
    return {"doctors": all_doctors}

@app.post("/api/precautions/appointment")
async def book_appointment(request: AppointmentRequest):
    """Book an appointment with a doctor"""
    try:
        # In a real system, this would save to a database
        # For now, we'll simulate a successful booking
        appointment_id = f"APT_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "message": f"Appointment booked successfully! Your appointment ID is {appointment_id}",
            "details": {
                "doctor_id": request.doctor_id,
                "patient_name": request.patient_name,
                "patient_email": request.patient_email,
                "patient_phone": request.patient_phone,
                "preferred_date": request.preferred_date,
                "preferred_time": request.preferred_time,
                "reason": request.reason,
                "status": "Confirmed",
                "booking_date": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Appointment booking failed: {str(e)}")

@app.get("/precautions", response_class=HTMLResponse)
async def precautions_page():
    """Serve the Precautions/AI Doctor page"""
    try:
        with open("frontend/precautions.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Precautions page not found"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.DEBUG
    )
