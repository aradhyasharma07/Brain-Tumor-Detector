from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.prediction_router import router as prediction_router

app = FastAPI(
    title="Brain Tumor MRI Classifier API",
    description="FastAPI Inference Server for ResNet50 Transfer-Learning Brain Tumor Detection",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(prediction_router)


@app.get("/")
def home():
    return {
        "message": "Brain Tumor Detection API is active",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "service": "Brain Tumor Detection FastAPI Server"
    }