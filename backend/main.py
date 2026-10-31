"""
CipherCare Backend: FastAPI + TenSEAL 0.3.18
Privacy-Preserving Medical Inference Server

This server performs encrypted inference on homomorphically encrypted patient data.
- Never receives or stores secret keys
- Never decrypts patient data
- Never accepts plaintext features
- Never returns plaintext predictions
"""

import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
import tenseal as ts
from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION AND LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/breast_cancer_pipeline.joblib"
EXPECTED_FEATURES = 30


# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Application state holder for model and configuration."""
    def __init__(self):
        self.pipeline = None
        self.classifier = None
        self.coef = None
        self.intercept = None
        self.model_loaded = False


app_state = AppState()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service_name: str = Field(..., description="Service identifier")
    model_loaded: bool = Field(..., description="Model loading status")


class InferenceRequest(BaseModel):
    """Encrypted inference request."""
    public_context: str = Field(
        ..., 
        description="Base64-encoded serialized public-only CKKS context"
    )
    encrypted_features: str = Field(
        ..., 
        description="Base64-encoded serialized CKKS encrypted vector (30 features)"
    )


class InferenceResponse(BaseModel):
    """Encrypted inference response."""
    encrypted_result: str = Field(
        ..., 
        description="Base64-encoded serialized CKKS encrypted result (log-odds score)"
    )


# ============================================================================
# INITIALIZATION
# ============================================================================

def load_model():
    """
    Load the trained breast cancer pipeline at application startup.
    
    Raises:
        FileNotFoundError: If model file not found
        Exception: If model structure is invalid
    """
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    try:
        pipeline = load(MODEL_PATH)
        
        # Verify pipeline structure
        if not hasattr(pipeline, 'named_steps'):
            raise ValueError("Pipeline missing 'named_steps' attribute")
        
        if 'classifier' not in pipeline.named_steps:
            raise ValueError("Pipeline missing 'classifier' step")
        
        classifier = pipeline.named_steps['classifier']
        
        # Extract coefficients and intercept
        coef = classifier.coef_[0]  # Shape: (30,)
        intercept = classifier.intercept_[0]
        
        if len(coef) != EXPECTED_FEATURES:
            raise ValueError(
                f"Model has {len(coef)} features, expected {EXPECTED_FEATURES}"
            )
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Coefficients shape: {coef.shape}")
        logger.info(f"  Intercept: {intercept:.6f}")
        
        return pipeline, classifier, coef, intercept
    
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {MODEL_PATH}")
        raise
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Load model on startup, cleanup on shutdown.
    """
    # Startup
    logger.info("=" * 80)
    logger.info("CipherCare Backend Starting")
    logger.info("=" * 80)
    
    try:
        pipeline, classifier, coef, intercept = load_model()
        app_state.pipeline = pipeline
        app_state.classifier = classifier
        app_state.coef = coef
        app_state.intercept = intercept
        app_state.model_loaded = True
        logger.info("✓ Application startup complete")
    except Exception as e:
        logger.error(f"✗ Failed to start application: {e}")
        app_state.model_loaded = False
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("CipherCare Backend Shutting Down")
    logger.info("=" * 80)
    app_state.model_loaded = False
    logger.info("✓ Shutdown complete")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="CipherCare Backend",
    description="Privacy-Preserving Medical Inference using Homomorphic Encryption",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Service status, name, and model loading status
    
    Example response:
        {
            "status": "healthy",
            "service_name": "CipherCare Backend",
            "model_loaded": true
        }
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy" if app_state.model_loaded else "degraded",
        service_name="CipherCare Backend",
        model_loaded=app_state.model_loaded
    )


@app.post("/infer", response_model=InferenceResponse)
async def encrypted_inference(request: InferenceRequest):
    """
    Perform encrypted inference on homomorphically encrypted patient data.
    
    This endpoint:
    1. Receives base64-encoded serialized public context and encrypted features
    2. Deserializes the public CKKS context (no secret key involved)
    3. Loads the encrypted feature vector
    4. Performs encrypted dot product with model coefficients
    5. Adds plaintext intercept to encrypted result
    6. Serializes and returns encrypted result
    
    Security guarantees:
    - No secret key is received, stored, or used
    - Patient data remains encrypted throughout computation
    - No plaintext features or predictions are returned
    - Server cannot decrypt results without client's secret key
    
    Args:
        request: InferenceRequest containing:
            - public_context: Base64-encoded public-only CKKS context
            - encrypted_features: Base64-encoded encrypted feature vector
    
    Returns:
        InferenceResponse: Base64-encoded encrypted result (log-odds score)
    
    Raises:
        HTTPException 400: If model not loaded or input validation fails
        HTTPException 422: If base64 decoding fails
        HTTPException 500: If computation fails
    
    Example request:
        {
            "public_context": "base64_encoded_context_bytes",
            "encrypted_features": "base64_encoded_vector_bytes"
        }
    
    Example response:
        {
            "encrypted_result": "base64_encoded_result_bytes"
        }
    """
    logger.info("Inference request received")
    
    # Verify model is loaded
    if not app_state.model_loaded:
        logger.error("Model not loaded, cannot process inference")
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Service is not ready for inference."
        )
    
    # Decode base64 inputs
    logger.info("Decoding base64 inputs...")
    try:
        context_public_bytes = base64.b64decode(request.public_context)
        x_encrypted_bytes = base64.b64decode(request.encrypted_features)
        logger.info(f"  Public context size: {len(context_public_bytes)} bytes")
        logger.info(f"  Encrypted features size: {len(x_encrypted_bytes)} bytes")
    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Base64 decoding failed: {str(e)}"
        )
    
    # Reconstruct public-only context
    logger.info("Deserializing public-only context...")
    try:
        context_public = ts.context_from(context_public_bytes)
        logger.info("✓ Public context deserialized successfully")
    except Exception as e:
        logger.error(f"Context deserialization failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to deserialize context: {str(e)}"
        )
    
    # Load encrypted features
    logger.info("Loading encrypted features...")
    try:
        x_encrypted = ts.ckks_vector_from(context_public, x_encrypted_bytes)
        logger.info("✓ Encrypted features loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load encrypted features: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to load encrypted features: {str(e)}"
        )
    
    # Perform encrypted dot product
    logger.info("Performing encrypted dot product...")
    try:
        result_encrypted = x_encrypted.dot(app_state.coef)
        logger.info("✓ Encrypted dot product completed")
    except Exception as e:
        logger.error(f"Dot product computation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Encrypted dot product failed: {str(e)}"
        )
    
    # Add intercept (plaintext value added to encrypted result)
    logger.info(f"Adding intercept ({app_state.intercept:.6f}) to encrypted result...")
    try:
        result_encrypted += app_state.intercept
        logger.info("✓ Intercept added successfully")
    except Exception as e:
        logger.error(f"Adding intercept failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Adding intercept failed: {str(e)}"
        )
    
    # Serialize encrypted result
    logger.info("Serializing encrypted result...")
    try:
        result_encrypted_bytes = result_encrypted.serialize()
        logger.info(f"✓ Result serialized ({len(result_encrypted_bytes)} bytes)")
    except Exception as e:
        logger.error(f"Result serialization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to serialize result: {str(e)}"
        )
    
    # Encode to base64
    logger.info("Encoding result to base64...")
    try:
        result_base64 = base64.b64encode(result_encrypted_bytes).decode('utf-8')
        logger.info("✓ Inference completed successfully")
    except Exception as e:
        logger.error(f"Base64 encoding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Base64 encoding failed: {str(e)}"
        )
    
    return InferenceResponse(encrypted_result=result_base64)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Log HTTP exceptions."""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return {
        "detail": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Log unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "detail": "Internal server error",
        "status_code": 500
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting CipherCare FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
