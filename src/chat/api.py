import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from ..services.coordinator import SpendSenseCoordinator

app = FastAPI(
    title="SpendSense API",
    description="Backend API for SpendSense Personal Finance Coach",
    version="1.0.0"
)

# Enable CORS for external integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize single instance coordinator
coordinator = SpendSenseCoordinator()

@app.post("/api/chat")
def chat_endpoint(payload: Dict[str, str]):
    """Send text message to financial coach."""
    message = payload.get("message")
    session_id = payload.get("session_id", "default_api_session")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message field is required.")
        
    try:
        reply = coordinator.process_text_message(message, session_id)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-receipt")
async def upload_receipt_endpoint(file: UploadFile = File(...), session_id: str = "default_api_session"):
    """Upload payment screenshot for OCR transaction extraction."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        reply = coordinator.process_image_upload(temp_path, session_id)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@app.post("/api/upload-audio")
async def upload_audio_endpoint(file: UploadFile = File(...), session_id: str = "default_api_session"):
    """Upload audio voice recording for speech-to-text and coaching response."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name
        
    try:
        reply = coordinator.process_voice_recording(temp_path, session_id)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

@app.get("/api/transactions")
def get_transactions():
    """Retrieve all logged transactions."""
    txs = coordinator.get_all_transactions()
    return [{
        "id": t.id,
        "merchant": t.merchant,
        "category": t.category,
        "amount": t.amount,
        "date": t.date,
        "raw_text": t.raw_text,
        "ocr_source": t.ocr_source,
        "created_at": t.created_at
    } for t in txs]

@app.delete("/api/transactions/{tx_id}")
def delete_transaction(tx_id: int):
    """Delete a transaction by ID."""
    success = coordinator.delete_transaction(tx_id)
    if not success:
        raise HTTPException(status_code=404, detail="Transaction not found or could not be deleted.")
    return {"message": f"Transaction {tx_id} deleted successfully."}

@app.post("/api/summary/{period_type}")
def generate_summary(period_type: str):
    """Generate weekly or monthly summary."""
    if period_type not in ["weekly", "monthly"]:
        raise HTTPException(status_code=400, detail="Invalid period type. Use 'weekly' or 'monthly'.")
        
    res = coordinator.generate_periodic_summary(period_type)
    if res["status"] == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res
