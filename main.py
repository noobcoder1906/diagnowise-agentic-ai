from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import redis
import json
import firebase_admin
from firebase_admin import credentials, auth
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, EmailStr
import base64
from typing import List, Optional
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import uuid
import io

from SymptomAgent.agents import create_symptom_checker_agent
from SymptomAgent.task import create_diagnosis_task
from SymptomAgent.tools import get_diseases_from_neo4j, close_driver
from HistoryAgent.agents import medical_history_agent
from HistoryAgent.task import create_history_analysis_task
from HistoryAgent.pdf_generator import generate_medical_report_pdf_memory
from EmergencyAgent.agents import emergency_agent
from EmergencyAgent.tasks import create_firstaid_task

from crewai import Crew
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
import asyncio
import tempfile
from datetime import datetime, timedelta

from appointment.crew import EnhancedHealthcareCrewAI, HEALTHCARE_PROVIDERS

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin SDK
# Replace with your service account key path
cred = credentials.Certificate("service.json")
firebase_admin.initialize_app(cred)


# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),  # Your Redis host
    port=13590,  # Your Redis port
    password=os.getenv("REDIS_PASSWORD"),  # Your Redis password
    decode_responses=True
)

# Security
security = HTTPBearer()

# LLM initialization
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models
class SymptomRequest(BaseModel):
    symptoms: list[str]

class MessageRequest(BaseModel):
    symptoms: str

class ReportRequest(BaseModel):
    history: str
    patient_name: Optional[str] = "Patient"

class EmergencyRequest(BaseModel):
    emergency: str

class TranscriptionResponse(BaseModel):
    transcription: str

class EnhancedPatientData(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history reported."
    appointment_type: Optional[str] = "consultation"
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class EnhancedMedicalAnalysisRequest(BaseModel):
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history."
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class AppointmentRequest(BaseModel):
    patient_id: str
    provider_name: str
    date: str
    time: str
    appointment_type: str
    reason: Optional[str] = ""

# Custom Redis Chat Message History with user-specific sessions
class UserRedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, user_id: str, session_id: str = "default"):
        self.user_id = user_id
        self.session_id = session_id
        self.key = f"chat_history:{user_id}:{session_id}"
        self.redis_client = redis_client

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from Redis"""
        try:
            messages_json = self.redis_client.lrange(self.key, 0, -1)
            messages = []
            for msg_json in messages_json:
                try:
                    msg_dict = json.loads(msg_json)
                    # Convert single message dict to list format expected by messages_from_dict
                    converted_messages = messages_from_dict([msg_dict])
                    messages.extend(converted_messages)
                except Exception as e:
                    print(f"Error parsing message: {e}")
                    continue
            return messages
        except Exception as e:
            print(f"Error retrieving messages: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to Redis"""
        try:
            message_dict = message_to_dict(message)
            message_json = json.dumps(message_dict)
            self.redis_client.rpush(self.key, message_json)
            # Set expiration (optional) - 30 days
            self.redis_client.expire(self.key, 30 * 24 * 60 * 60)
            print(f"Message added to Redis key: {self.key}")  # Add logging
        except Exception as e:
            print(f"Error adding message to Redis: {e}")

    def clear(self) -> None:
        """Clear chat history"""
        try:
            self.redis_client.delete(self.key)
            print(f"Chat history cleared for key: {self.key}")
        except Exception as e:
            print(f"Error clearing chat history: {e}")


# Firebase Auth verification
async def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Extract token from "Bearer <token>"
        token = credentials.credentials
        
        # Verify the Firebase token
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        user_email = decoded_token.get('email', '')
        
        return {
            "user_id": user_id,
            "email": user_email,
            "decoded_token": decoded_token
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")

# Create session function for each user
def create_user_session(user_id: str, session_id: str = "default") -> BaseChatMessageHistory:
    return UserRedisChatMessageHistory(user_id=user_id, session_id=session_id)


# Symptom analysis route (with optional auth)
@app.post("/analyze-symptoms")
async def analyze_symptoms(data: SymptomRequest):
    try:
        agent = create_symptom_checker_agent(os.getenv("OPENAI_API_KEY"))
        diseases = get_diseases_from_neo4j(data.symptoms, top_n=5)
        task = create_diagnosis_task(agent, data.symptoms, diseases)
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        close_driver()
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

# Chatbot route with user-specific memory
@app.post("/chatbot")
async def chat(message: MessageRequest, user_data: dict = Depends(verify_firebase_token)):
    try:
        user_id = user_data["user_id"]
        session_id = f"chat_{user_id}"
        
        # Create user-specific session
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            return UserRedisChatMessageHistory(user_id=user_id, session_id="default")
        
        # Create LLM with user-specific message history
        llm_history = RunnableWithMessageHistory(llm, get_session_history)
        
        # Configuration with user-specific session ID
        config = {"configurable": {"session_id": session_id}}

        # Create the system message and user message
        system_message = SystemMessage(
            content="""
You are VitalLens, a trusted and empathetic AI health assistant trained to provide reliable information and guidance on medical symptoms, wellness, and healthcare.
Your role is a blend of a knowledgeable doctor and a caring nurse.

Your objectives:
- Assist users in understanding symptoms, conditions, medications, and general wellness.
- Offer actionable, concise explanations tailored to the user's concern or level of understanding.
- Clearly communicate when the user should seek professional in-person care.
- Be empathetic, non-judgmental, and supportive in tone.
- Only respond to questions strictly related to physical or mental health, medicine, wellness, or healthcare practices.

Rules you must follow:
- Do not engage with questions unrelated to health or medicine.
- Never provide a formal diagnosis or prescribe medication.
- Avoid speculative or experimental medical advice.
- Always include a disclaimer suggesting consulting a healthcare provider for serious concerns.
- If a query is vague or unclear, politely ask for more details.
- Keep responses informative, calm, and medically grounded.

Example personas you represent:
- A nurse: reassuring, practical, patient-focused
- A doctor: precise, medically knowledgeable, professional

Always keep the conversation in the healthcare domain.

NOTE: You have memory of our previous conversations to provide personalized care.
"""
        )
        
        # IMPORTANT: Use invoke with just the user message, not the system message
        # The system message should be added to the chat history manually if needed
        user_message = HumanMessage(content=message.symptoms)
        
        # Manually add system message to history if this is the first interaction
        chat_history = get_session_history(session_id)
        if not chat_history.messages:
            chat_history.add_message(system_message)
        
        # Add user message to history
        chat_history.add_message(user_message)
        
        # Invoke the LLM with message history
        result = llm_history.invoke(
            [user_message],  # Only pass the current user message
            config=config
        )
        
        # The response should be automatically added to history by RunnableWithMessageHistory
        return {"response": result.content}

    except Exception as e:
        print(f"Chat error: {str(e)}")  # Add logging
        raise HTTPException(status_code=500, detail=f"Failed to generate a response: {str(e)}")
    

# Report route with auth
@app.post("/report")
async def report(history: ReportRequest):
    try:
        # Generate the medical report using CrewAI
        agent = medical_history_agent
        history_task = create_history_analysis_task(history.history, medical_history_agent)
        crew = Crew(agents=[agent], tasks=[history_task], verbose=True)
        result = crew.kickoff()
        
        print(f"\nDEBUG - Raw result type: {type(result)}")
        print(f"DEBUG - Raw result content: {result}")
        
        # Get patient name
        patient_name = getattr(history, 'patient_name', 'Patient')
        
        # Generate PDF in memory and get bytes + filename
        pdf_bytes, pdf_filename = generate_medical_report_pdf_memory(result, patient_name)
        
        print(f"\n✅ PDF Report generated successfully in memory: {pdf_filename}")
        
        # Convert PDF bytes to base64 for JSON response
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Return the analysis result and PDF data
        return {
            "success": True,
            "analysis": str(result),
            "pdf_filename": pdf_filename,
            "pdf_data": pdf_base64,
            "pdf_size": len(pdf_bytes)
        }
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate medical report: {str(e)}")

@app.post("/report-pdf")
async def report_pdf(history: ReportRequest):
    try:
        # Generate the medical report using CrewAI
        agent = medical_history_agent
        history_task = create_history_analysis_task(history.history, medical_history_agent)
        crew = Crew(agents=[agent], tasks=[history_task], verbose=True)
        result = crew.kickoff()
        
        # Get patient name
        patient_name = getattr(history, 'patient_name', 'Patient')
        
        # Generate PDF in memory
        pdf_bytes, pdf_filename = generate_medical_report_pdf_memory(result, patient_name)
        
        print(f"\n✅ PDF Report generated successfully in memory: {pdf_filename}")
        
        # Return the PDF as a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={pdf_filename}",
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate medical report: {str(e)}")

# Optional: Separate endpoint just for downloading if you already have the analysis
@app.post("/download-pdf")
async def download_pdf(request: dict):
    try:
        # Extract analysis and patient name from request
        analysis = request.get('analysis', '')
        patient_name = request.get('patient_name', 'Patient')
        
        if not analysis:
            raise HTTPException(status_code=400, detail="Analysis data is required")
        
        # Generate PDF from existing analysis
        pdf_bytes, pdf_filename = generate_medical_report_pdf_memory(analysis, patient_name)
        
        print(f"\n✅ PDF downloaded successfully: {pdf_filename}")
        
        # Return the PDF as a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={pdf_filename}",
                "Content-Length": str(len(pdf_bytes))
            }
        )
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")

@app.post("/emergency")
async def emergency(message: EmergencyRequest):
    first_aid_task = create_firstaid_task(message.emergency)
    firstaid_crew = Crew(
        agents=[emergency_agent],
        tasks=[first_aid_task],
        verbose=True
    )
    results = firstaid_crew.kickoff()
    return {"result": str(results)}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Validate file type
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read the uploaded file
        audio_data = await audio.read()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Use OpenAI Whisper API for transcription
            import openai
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return TranscriptionResponse(transcription=transcript)
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/voice_emergency_report")
async def voice_emergency_report(audio: UploadFile = File(...)):
    try:
        # Validate file type
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read the uploaded file
        audio_data = await audio.read()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Use OpenAI Whisper API for transcription
            import openai
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            # Now call the emergency agent with the transcribed text
            first_aid_task = create_firstaid_task(transcript)
            firstaid_crew = Crew(
                agents=[emergency_agent],
                tasks=[first_aid_task],
                verbose=True
            )
            results = firstaid_crew.kickoff()
            
            return {
                "transcription": transcript,
                "emergency_guidance": str(results)
            }
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice emergency report failed: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Test Redis connection
        redis_client.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": f"error: {str(e)}"}

# Clear chat history endpoint (optional)
@app.delete("/chat/clear")
async def clear_chat_history(user_data: dict = Depends(verify_firebase_token)):
    try:
        user_id = user_data["user_id"]
        chat_history = UserRedisChatMessageHistory(user_id=user_id)
        chat_history.clear()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear chat history: {str(e)}")

# Get chat history endpoint (optional)
@app.get("/chat/history")
async def get_chat_history(user_data: dict = Depends(verify_firebase_token)):
    try:
        user_id = user_data["user_id"]
        chat_history = UserRedisChatMessageHistory(user_id=user_id)
        messages = chat_history.messages
        
        # Convert messages to a serializable format
        serialized_messages = []
        for msg in messages:
            serialized_messages.append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            })
        
        return {"messages": serialized_messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")


healthcare_system = EnhancedHealthcareCrewAI()

# Enhanced in-memory storage
appointments_db = {}
patients_db = {}
reports_db = {}
email_logs = {}

# Enhanced Pydantic models
class EnhancedPatientData(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history reported."
    appointment_type: Optional[str] = "consultation"
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class EnhancedMedicalAnalysisRequest(BaseModel):
    symptoms: List[str]
    medical_history: Optional[str] = "No significant medical history."
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    urgency_level: Optional[str] = "routine"

class AppointmentRequest(BaseModel):
    patient_id: str
    provider_name: str
    date: str
    time: str
    appointment_type: str
    reason: Optional[str] = ""

# Enhanced API Endpoints

@app.get("/")
async def root():
    return {
        "message": "Enhanced Healthcare AI API with Auto-Email is running",
        "version": "2.0.0",
        "features": [
            "Comprehensive AI Medical Analysis",
            "Automatic Email Delivery",
            "Date/Time Preferences",
            "All CrewAI Agents Utilized",
            "Enhanced PDF Reports"
        ],
        "endpoints": [
            "/providers",
            "/appointments",
            "/process-patient-enhanced",
            "/medical-analysis-enhanced",
            "/email-logs"
        ]
    }

@app.get("/providers")
async def get_enhanced_providers(specialty: Optional[str] = None):
    """Get enhanced healthcare providers with availability"""
    try:
        providers_list = []
        for spec, provider in HEALTHCARE_PROVIDERS.items():
            if specialty and specialty.lower() not in spec.lower():
                continue
                
            providers_list.append({
                "id": spec,
                "name": provider["name"],
                "specialty": spec.replace("_", " ").title(),
                "email": provider["email"],
                "location": provider["location"],
                "specializations": provider["specializations"],
                "available_slots": provider.get("available_slots", []),
                "rating": 4.8,
                "nextAvailable": "Next Week",
                "experience": "15+ years"
            })
        
        return {"providers": providers_list, "total": len(providers_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-patient-enhanced")
async def process_patient_enhanced(patient_data: EnhancedPatientData, background_tasks: BackgroundTasks):
    """Enhanced patient processing with automatic email delivery"""
    try:
        patient_id = str(uuid.uuid4())
        
        # Convert to dict for processing
        patient_dict = {
            "name": patient_data.name,
            "email": patient_data.email,
            "phone": patient_data.phone or "Not provided",
            "symptoms": patient_data.symptoms,
            "medical_history": patient_data.medical_history,
            "preferred_date": patient_data.preferred_date,
            "preferred_time": patient_data.preferred_time,
            "urgency_level": patient_data.urgency_level
        }
        
        # Store patient data
        patients_db[patient_id] = patient_dict
        
        print(f"🚀 Enhanced processing started for: {patient_data.name}")
        
        # Process in background with auto-email
        background_tasks.add_task(process_patient_enhanced_background, patient_id, patient_dict)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "message": "Enhanced AI processing started. Comprehensive medical report will be emailed automatically.",
            "status": "processing",
            "features_used": [
                "Medical History Analyst Agent",
                "Symptom Diagnostician Agent", 
                "Appointment Coordinator Agent",
                "Automatic Email Delivery",
                "Comprehensive PDF Report"
            ]
        }
        
    except Exception as e:
        print(f"❌ Enhanced processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

async def process_patient_enhanced_background(patient_id: str, patient_data: dict):
    """Enhanced background processing with all CrewAI agents"""
    try:
        print(f"🤖 Starting enhanced background processing for: {patient_id}")
        
        # Process through enhanced crew system
        results = healthcare_system.process_patient_with_auto_email(patient_data)
        
        # Store comprehensive results
        if results['success']:
            appointments_db[patient_id] = {
                "patient_id": patient_id,
                "appointment_details": results['appointment_details'],
                "medical_analysis": results['medical_history_analysis'],
                "clinical_assessment": results['clinical_assessment'],
                "appointment_coordination": results['appointment_coordination'],
                "pdf_report_path": results.get('pdf_report_path'),
                "urgency": results.get('urgency', 'routine'),
                "status": "confirmed",
                "email_sent": results.get('email_sent', False),
                "email_status": results.get('email_status', ''),
                "processing_summary": results.get('processing_summary', ''),
                "created_at": datetime.now().isoformat()
            }
            
            # Store report info
            reports_db[patient_id] = {
                "patient_id": patient_id,
                "report_path": results.get('pdf_report_path'),
                "generated_at": datetime.now().isoformat(),
                "email_delivered": results.get('email_sent', False)
            }
            
            # Log email delivery
            email_logs[patient_id] = {
                "patient_id": patient_id,
                "patient_email": patient_data['email'],
                "email_sent": results.get('email_sent', False),
                "email_status": results.get('email_status', ''),
                "timestamp": datetime.now().isoformat()
            }
            
        print(f"✅ Enhanced background processing completed for: {patient_id}")
        
    except Exception as e:
        print(f"❌ Enhanced background processing failed: {str(e)}")
        # Store error info
        appointments_db[patient_id] = {
            "patient_id": patient_id,
            "status": "failed",
            "error": str(e),
            "created_at": datetime.now().isoformat()
        }

@app.post("/medical-analysis-enhanced")
async def get_enhanced_medical_analysis(request: EnhancedMedicalAnalysisRequest):
    """Enhanced medical analysis with date/time preferences"""
    try:
        # Create enhanced temporary patient data
        temp_patient = {
            "name": "Analysis Request",
            "email": "temp@example.com",
            "symptoms": request.symptoms,
            "medical_history": request.medical_history,
            "preferred_date": request.preferred_date,
            "preferred_time": request.preferred_time,
            "urgency_level": request.urgency_level
        }
        
        # Run enhanced analysis through history agent
        history_agent = healthcare_system.create_enhanced_medical_history_agent()
        history_task = healthcare_system.create_comprehensive_medical_analysis_task(history_agent, temp_patient)
        
        from crewai import Crew, Process
        history_crew = Crew(agents=[history_agent], tasks=[history_task], process=Process.sequential)
        analysis_result = history_crew.kickoff()
        
        # Determine recommended specialty and provider
        symptom_text = " ".join(request.symptoms).lower()
        best_specialty = "internal_medicine"
        
        for specialty, provider in HEALTHCARE_PROVIDERS.items():
            if any(spec in symptom_text for spec in provider['specializations']):
                best_specialty = specialty
                break
        
        recommended_provider = HEALTHCARE_PROVIDERS[best_specialty]
        
        # Consider date/time preferences in response
        scheduling_note = ""
        if request.preferred_date or request.preferred_time:
            scheduling_note = f"Preferences noted: Date: {request.preferred_date or 'Flexible'}, Time: {request.preferred_time or 'Flexible'}"
        
        return {
            "success": True,
            "analysis": str(analysis_result),
            "recommended_specialty": best_specialty.replace("_", " ").title(),
            "recommended_provider": recommended_provider,
            "urgency": request.urgency_level,
            "scheduling_preferences": scheduling_note,
            "available_slots": recommended_provider.get("available_slots", []),
            "analysis_features": [
                "Risk factor assessment",
                "Medication alerts",
                "Differential diagnosis",
                "Clinical correlation",
                "Urgency classification"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/appointments")
async def get_all_enhanced_appointments():
    """Get all appointments with enhanced details"""
    try:
        appointments_list = []
        for patient_id, appt in appointments_db.items():
            patient = patients_db.get(patient_id, {})
            appointment_details = appt.get('appointment_details', {})
            
            appointments_list.append({
                "id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "provider": appointment_details.get('doctor', 'TBD'),
                "specialty": appointment_details.get('specialty', 'General'),
                "date": appointment_details.get('date', 'TBD'),
                "time": appointment_details.get('time', 'TBD'),
                "type": "Enhanced AI Consultation",
                "location": appointment_details.get('location', 'TBD'),
                "status": appt.get('status', 'pending'),
                "urgency": appt.get('urgency', 'routine'),
                "email_sent": appt.get('email_sent', False),
                "email_status": appt.get('email_status', ''),
                "processing_summary": appt.get('processing_summary', ''),
                "created_at": appt.get('created_at', '')
            })
        
        return {
            "appointments": appointments_list,
            "total": len(appointments_list),
            "email_delivery_rate": sum(1 for a in appointments_list if a['email_sent']) / max(len(appointments_list), 1) * 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/email-logs")
async def get_email_logs():
    """Get email delivery logs"""
    try:
        logs_list = []
        for patient_id, log in email_logs.items():
            patient = patients_db.get(patient_id, {})
            logs_list.append({
                "patient_id": patient_id,
                "patient_name": patient.get('name', 'Unknown'),
                "patient_email": log['patient_email'],
                "email_sent": log['email_sent'],
                "email_status": log['email_status'],
                "timestamp": log['timestamp']
            })
        
        return {
            "email_logs": logs_list,
            "total_emails": len(logs_list),
            "successful_deliveries": sum(1 for log in logs_list if log['email_sent']),
            "delivery_rate": sum(1 for log in logs_list if log['email_sent']) / max(len(logs_list), 1) * 100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-slots")
async def get_enhanced_available_slots(date: str, provider: Optional[str] = None):
    """Get enhanced available time slots with provider-specific schedules"""
    try:
        # Get provider-specific slots if specified
        if provider:
            for specialty, prov_data in HEALTHCARE_PROVIDERS.items():
                if prov_data["name"] == provider:
                    base_slots = prov_data.get("available_slots", [
                        "9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"
                    ])
                    break
            else:
                base_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
        else:
            base_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
        
        # Filter out booked slots
        available_slots = base_slots.copy()
        
        for appt in appointments_db.values():
            appt_details = appt.get("appointment_details", {})
            if appt_details.get("date") == date and appt.get("status") == "confirmed":
                booked_time = appt_details.get("time")
                if booked_time in available_slots:
                    available_slots.remove(booked_time)
        
        return {
            "success": True,
            "date": date,
            "provider": provider,
            "available_slots": available_slots,
            "total_slots": len(available_slots)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{patient_id}")
async def get_enhanced_medical_report(patient_id: str):
    """Download enhanced medical report PDF"""
    try:
        if patient_id not in reports_db:
            raise HTTPException(status_code=404, detail="Enhanced medical report not found")
        
        report_info = reports_db[patient_id]
        pdf_path = report_info["report_path"]
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        return FileResponse(
            path=pdf_path,
            filename=f"comprehensive_medical_report_{patient_id}.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/health")
async def enhanced_health_check():
    """Enhanced system health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "patients_count": len(patients_db),
        "appointments_count": len(appointments_db),
        "reports_count": len(reports_db),
        "emails_sent": len(email_logs),
        "email_delivery_rate": sum(1 for log in email_logs.values() if log['email_sent']) / max(len(email_logs), 1) * 100,
        "features": [
            "Enhanced CrewAI Integration",
            "Automatic Email Delivery", 
            "Date/Time Preferences",
            "Comprehensive PDF Reports",
            "All AI Agents Utilized"
        ]
    }

@app.post("/system/reset")
async def reset_enhanced_system():
    """Reset all enhanced system data"""
    global appointments_db, patients_db, reports_db, email_logs
    appointments_db.clear()
    patients_db.clear()
    reports_db.clear()
    email_logs.clear()
    
    return {
        "success": True,
        "message": "Enhanced system data reset successfully",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Healthcare AI FastAPI Server...")
    print("✨ New Features:")
    print("   - Automatic Email Delivery")
    print("   - Date/Time Preferences in AI Analysis")
    print("   - All CrewAI Agents Utilized")
    print("   - Enhanced PDF Reports")
    print("   - Comprehensive Medical Analysis")
    print("\n📋 Enhanced Endpoints:")
    print("   - POST /process-patient-enhanced")
    print("   - POST /medical-analysis-enhanced") 
    print("   - GET  /email-logs")
    print("   - GET  /providers (enhanced)")
    print("   - GET  /appointments (enhanced)")
    print("\n🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "enhanced_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

@app.post("/emergency")
async def emergency(message: EmergencyRequest):
    first_aid_task = create_firstaid_task(message.emergency)
    firstaid_crew = Crew(
        agents=[emergency_agent],
        tasks=[first_aid_task],
        verbose=True
    )
    results = firstaid_crew.kickoff()
    return {"result": str(results)}


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import io
import tempfile
import os
from pydantic import BaseModel



@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        # Validate file type
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read the uploaded file
        audio_data = await audio.read()
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Use OpenAI Whisper API for transcription
            with open(temp_file_path, "rb") as audio_file:
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return TranscriptionResponse(transcription=transcript)
        
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    

