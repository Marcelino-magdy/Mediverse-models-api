# Multi-Model Medical AI API

## Project Structure

```
medical-ai-api/
├── main.py                           # Main FastAPI application
├── Dockerfile                        # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
├── requirements.txt                  # Python dependencies
├── nginx.conf                       # Nginx configuration (optional)
├── models/                          # Model files directory
│   ├── brain_mri_finetuned_model.h5
│   ├── chest_xray_model.h5
│   ├── bone_fracture_model.h5
│   ├── skin_cancer_model.h5
│   └── retinal_disease_model.h5
└── README.md                        # This file
```

## Available Models and Endpoints

| Model | Endpoint | Description |
|-------|----------|-------------|
| MRI Brain Tumor | `/prediction/mri-brain-tumor` | Classifies brain MRI images |
| X-Ray Chest | `/prediction/xray-chest` | Analyzes chest X-rays |
| Bone Fracture | `/prediction/bone-fracture` | Detects bone fractures |
| Skin Cancer | `/prediction/skin-cancer` | Classifies skin lesions |
| Retinal Disease | `/prediction/retinal-disease` | Detects retinal conditions |

## API Endpoints

### Core Endpoints
- `GET /` - API information and available models
- `GET /health` - Health check for all models
- `GET /models` - Detailed model information
- `POST /predict` - Legacy endpoint (MRI brain tumor)

### Prediction Endpoints
- `POST /prediction/mri-brain-tumor` - MRI brain tumor classification
- `POST /prediction/xray-chest` - Chest X-ray analysis
- `POST /prediction/bone-fracture` - Bone fracture detection
- `POST /prediction/skin-cancer` - Skin cancer classification
- `POST /prediction/retinal-disease` - Retinal disease detection

## Setup Instructions

### 1. Prepare Model Files
Create a `models/` directory and place your trained model files:
```bash
mkdir models
# Copy your model files to the models directory
cp brain_mri_finetuned_model.h5 models/
cp chest_xray_model.h5 models/
# ... add other models
```

### 2. Build and Run with Docker Compose
```bash
# Build and start the API
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# With nginx (production mode)
docker-compose --profile production up -d --build
```

### 3. Access the API
- API Documentation: http://localhost:8171/docs
- Health Check: http://localhost:8171/health
- Model List: http://localhost:8171/models

## Usage Examples

### Python Example
```python
import requests

# Upload image for MRI brain tumor prediction
with open('brain_scan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8171/prediction/mri-brain-tumor', files=files)
    result = response.json()
    print(f"Prediction: {result['predicted_label']} (Confidence: {result['confidence']:.2f})")

# Upload image for X-ray chest analysis
with open('chest_xray.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8171/prediction/xray-chest', files=files)
    result = response.json()
    print(f"Prediction: {result['predicted_label']} (Confidence: {result['confidence']:.2f})")
```

### cURL Example
```bash
# MRI Brain Tumor Prediction
curl -X POST "http://localhost:8171/prediction/mri-brain-tumor" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@brain_scan.jpg"

# Chest X-ray Prediction
curl -X POST "http://localhost:8171/prediction/xray-chest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg"
```

## Adding New Models

To add a new model:

1. **Add model configuration** in `main.py`:
```python
MODEL_CONFIGS = {
    # ... existing models
    "new-model": ModelConfig(
        name="New Model Description",
        model_path="models/new_model.h5",
        labels=["Label1", "Label2", "Label3"],
        img_size=(224, 224)
    )
}
```

2. **Add endpoint** in `main.py`:
```python
@app.post("/prediction/new-model")
async def predict_new_model(file: UploadFile = File(...)):
    """Predict using new model"""
    return await make_prediction("new-model", file)
```

3. **Place model file** in the `models/` directory

4. **Restart the service**:
```bash
docker-compose restart
```

## Model File Management

### Model Requirements
- Models should be in `.h5` format (Keras/TensorFlow)
- Input size should be (224, 224) for consistency
- Models should output class probabilities

### Model Preprocessing
The API uses ResNet50 preprocessing by default. You can modify the preprocessing by:
1. Adding new preprocessing functions in `get_preprocessing_function()`
2. Setting the `preprocessing_func` parameter in `ModelConfig`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python path | `/app` |
| `CUDA_VISIBLE_DEVICES` | GPU device | `0` |

## Monitoring and Logging

### Health Checks
- Docker health check every 30 seconds
- API health endpoint at `/health`
- Model loading status in logs

### Logs
View logs with:
```bash
docker-compose logs -f multi-model-medical-api
```

## Production Considerations

1. **Resource Requirements**:
   - Memory: 4GB+ recommended for multiple models
   - Storage: Depends on model sizes
   - CPU: Multi-core recommended

2. **Security**:
   - Implement authentication/authorization
   - Rate limiting
   - Input validation

3. **Scaling**:
   - Use nginx for load balancing
   - Consider model serving solutions like TensorFlow Serving
   - Implement caching for frequently used models

4. **Monitoring**:
   - Add metrics collection
   - Implement logging
   - Set up alerts for model failures