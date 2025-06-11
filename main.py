from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from keras.models import load_model
from PIL import Image
import io
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for each AI model"""
    name: str
    model_path: str
    labels: List[str]
    img_size: Tuple[int, int]
    preprocessing_func: str = "resnet50"  # preprocessing method

# Model configurations
MODEL_CONFIGS = {
    "mri-brain-tumor": ModelConfig(
        name="MRI Brain Tumor Classification",
        model_path="models/brain_mri_finetuned_model.h5",
        labels=["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        img_size=(224, 224)
    ),
    "xray-chest": ModelConfig(
        name="X-Ray Chest Classification",
        model_path="models/chest_xray_model.h5",
        labels=["Normal", "Pneumonia", "COVID-19", "Tuberculosis"],
        img_size=(224, 224)
    ),
    "bone-fracture": ModelConfig(
        name="Bone Fracture Detection",
        model_path="models/bone_fracture_model.h5",
        labels=["No Fracture", "Fracture"],
        img_size=(224, 224)
    ),
    "skin-cancer": ModelConfig(
        name="Skin Cancer Classification",
        model_path="models/skin_cancer_model.h5",
        labels=["Benign", "Malignant"],
        img_size=(224, 224)
    ),
    "retinal-disease": ModelConfig(
        name="Retinal Disease Detection",
        model_path="models/retinal_disease_model.h5",
        labels=["Normal", "Diabetic Retinopathy", "Glaucoma", "Macular Degeneration"],
        img_size=(224, 224)
    )
}

app = FastAPI(
    title="Multi-Model Medical AI API",
    description="API for various medical image classifications including MRI, X-Ray, and more",
    version="2.0.0"
)

# Global model storage
loaded_models: Dict[str, object] = {}

def load_all_models():
    """Load all available models at startup"""
    global loaded_models
    
    for model_key, config in MODEL_CONFIGS.items():
        try:
            if os.path.exists(config.model_path):
                model = load_model(config.model_path)
                loaded_models[model_key] = model
                print(f"✅ {config.name} model loaded successfully")
            else:
                print(f"⚠️  Model file not found: {config.model_path}")
                loaded_models[model_key] = None
        except Exception as e:
            print(f"❌ Error loading {config.name} model: {e}")
            loaded_models[model_key] = None

def get_preprocessing_function(preprocessing_type: str):
    """Get the appropriate preprocessing function"""
    if preprocessing_type == "resnet50":
        return preprocess_input
    # Add more preprocessing functions as needed
    return preprocess_input

def preprocess_image(image_bytes: bytes, config: ModelConfig):
    """Preprocess image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(config.img_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Apply preprocessing
    preprocessing_func = get_preprocessing_function(config.preprocessing_func)
    image_array = preprocessing_func(image_array)
    
    return image_array

# Initialize models at startup
@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    load_all_models()

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "message": "Multi-Model Medical AI API is running",
        "available_models": list(MODEL_CONFIGS.keys()),
        "endpoints": [f"/prediction/{model_key}" for model_key in MODEL_CONFIGS.keys()]
    }

@app.get("/health")
async def health_check():
    """Detailed health check for all models"""
    model_status = {}
    loaded_count = 0
    
    for model_key, config in MODEL_CONFIGS.items():
        is_loaded = loaded_models.get(model_key) is not None
        model_status[model_key] = {
            "name": config.name,
            "loaded": is_loaded,
            "labels": config.labels,
            "endpoint": f"/prediction/{model_key}"
        }
        if is_loaded:
            loaded_count += 1
    
    return {
        "status": "healthy" if loaded_count > 0 else "unhealthy",
        "total_models": len(MODEL_CONFIGS),
        "loaded_models": loaded_count,
        "models": model_status
    }

@app.get("/models")
async def list_models():
    """List all available models and their information"""
    models_info = {}
    
    for model_key, config in MODEL_CONFIGS.items():
        models_info[model_key] = {
            "name": config.name,
            "labels": config.labels,
            "input_size": config.img_size,
            "endpoint": f"/prediction/{model_key}",
            "available": loaded_models.get(model_key) is not None
        }
    
    return {"models": models_info}

async def make_prediction(model_key: str, file: UploadFile):
    """Generic prediction function for any model"""
    
    # Check if model key exists
    if model_key not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_key}' not found. Available models: {list(MODEL_CONFIGS.keys())}"
        )
    
    # Check if model is loaded
    model = loaded_models.get(model_key)
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model '{model_key}' is not available or failed to load"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        config = MODEL_CONFIGS[model_key]
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes, config)
        prediction = model.predict(image_array)
        
        class_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        predicted_label = config.labels[class_index]
        
        return {
            "model": config.name,
            "predicted_class": class_index,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                config.labels[i]: float(prediction[0][i]) 
                for i in range(len(config.labels))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dynamic endpoint creation for each model
@app.post("/prediction/mri-brain-tumor")
async def predict_mri_brain_tumor(file: UploadFile = File(...)):
    """Predict MRI brain tumor classification"""
    return await make_prediction("mri-brain-tumor", file)

@app.post("/prediction/xray-chest")
async def predict_xray_chest(file: UploadFile = File(...)):
    """Predict chest X-ray classification"""
    return await make_prediction("xray-chest", file)

@app.post("/prediction/bone-fracture")
async def predict_bone_fracture(file: UploadFile = File(...)):
    """Predict bone fracture detection"""
    return await make_prediction("bone-fracture", file)

@app.post("/prediction/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):
    """Predict skin cancer classification"""
    return await make_prediction("skin-cancer", file)

@app.post("/prediction/retinal-disease")
async def predict_retinal_disease(file: UploadFile = File(...)):
    """Predict retinal disease detection"""
    return await make_prediction("retinal-disease", file)

# Legacy endpoint for backward compatibility
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """Legacy endpoint - redirects to MRI brain tumor prediction"""
    return await make_prediction("mri-brain-tumor", file)