# Helmet Detection API

A FastAPI-based helmet detection system using YOLO (You Only Look Once) model with Google Drive integration for automatic result storage.

## 🚀 Features

- **Real-time Helmet Detection**: Uses YOLO model to detect people with and without helmets
- **Image & Video Processing**: Supports both image and video file analysis
- **Google Drive Integration**: Automatically uploads detection results to Google Drive
- **Object Tracking**: Optional tracking mode for video analysis
- **Configurable Parameters**: Adjustable confidence threshold, frame stride, and detection criteria
- **RESTful API**: Clean FastAPI endpoints with comprehensive documentation
- **CORS Support**: Cross-origin resource sharing enabled for web applications

## 📁 Project Structure

```
helmet-detection-api/
├── app/
│   └── main.py                 # Main FastAPI application
├── credentials/
│   ├── oauth-client.json       # Google OAuth credentials
│   └── token.pickle           # Cached OAuth token
├── outputs/                    # Detection result images
│   └── [video-name]-[id]/     # Organized by video name and request ID
├── runs/
│   └── detect/
│       ├── train/             # Training results and weights
│       └── train2/            # Additional training runs
├── tmp/                       # Temporary file storage
├── weights/
│   └── best.pt                # Trained YOLO model weights
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Dependencies

### Core Dependencies

- **FastAPI** (^0.104.1) - Modern, fast web framework for building APIs
- **Uvicorn** (^0.24.0) - ASGI server for FastAPI
- **Pydantic** (^2.5.0) - Data validation using Python type annotations
- **Python-dotenv** (^1.0.0) - Load environment variables from .env file
- **Python-multipart** (^0.0.6) - Support for multipart/form-data

### Computer Vision & ML

- **OpenCV** (^4.8.1) - Computer vision library for image/video processing
- **Ultralytics** (^8.0.196) - YOLO implementation for object detection
- **LAPX** (^0.5.12) - Linear Assignment Problem solver for tracking

### Google Drive Integration

- **Google API Python Client** (^2.108.0) - Google APIs client library
- **Google Auth** (^2.23.4) - Google authentication library
- **Google Auth HTTPLib2** (^0.2.0) - HTTP transport for Google Auth
- **Google Auth OAuthLib** (^0.1.1) - OAuth 2.0 flow for Google APIs

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Wattanaroj2567/Mini-Projet-ML.git
cd Mini-Projet-ML

# Install dependencies
pip install -r requirements.txt
```

### 2. Google Drive Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download the credentials JSON file
6. Place it in `credentials/oauth-client.json`

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_PATH=weights/best.pt
CONF_DEFAULT=0.35
LOG_LEVEL=info

# Output Configuration
OUTPUTS_DIR=outputs
TMP_DIR=tmp
OUTPUT_RETENTION_DAYS=7

# Google Drive Configuration
GDRIVE_ENABLED=1
GDRIVE_CREDENTIALS_PATH=credentials/oauth-client.json
GDRIVE_TOKEN_PATH=credentials/token.pickle
GDRIVE_PARENT_FOLDER_ID=your_folder_id_here
GDRIVE_AUTH_PORT=8080

# Frame Processing
MAX_SAVED_FRAMES=20
FRAME_STRIDE=5
PREVIEW_ONLY_ONE=1
PREVIEW_STRATEGY=max_unhelmet_then_conf

# Frame Saving Policy
SAVE_ONLY_MIXED=0
SAVE_FRAMES_WITH_HELMET=1
SAVE_FRAMES_WITHOUT_HELMET=1

# CORS
ALLOW_ORIGINS=http://localhost:3000,http://localhost:8080
```

### 4. Run the Application

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or with specific configuration
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

## 📚 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /
```

Returns API status and configuration information.

#### 2. Health Status

```http
GET /analyze/helmet-detection/health
```

Returns detailed health status including model and Google Drive status.

#### 3. Helmet Detection

```http
POST /analyze/helmet-detection
```

**Parameters:**

- `file` (file): Image or video file to analyze
- `conf` (float, optional): Confidence threshold (default: 0.35)
- `frame_stride` (int, optional): Frame sampling interval (default: 5)
- `max_frames` (int, optional): Maximum frames to process (default: 300)
- `track` (int, optional): Enable tracking mode (0=disabled, 1=enabled)
- `imgsz` (int, optional): Input image size (default: 640)
- `tracker_name` (str, optional): Tracker configuration (default: "botsort.yaml")
- `min_track_frames` (int, optional): Minimum frames for tracking (default: 2)
- `min_box_area` (int, optional): Minimum bounding box area (default: 0)
- `count_rule` (str, optional): Counting rule for tracking (default: "any-no-helmet")

**Response:**

```json
{
  "request_id": "abc12345",
  "source_type": "video",
  "total_person": 150,
  "count_helmet": 120,
  "count_no_helmet": 30,
  "confidence_threshold": 0.35,
  "per_frame": [...],
  "artifacts": {
    "gdrive_folder_id": "1ABC...",
    "gdrive_folder_url": "https://drive.google.com/...",
    "gdrive_uploaded": 15,
    "gdrive_total": 15,
    "gdrive_success": true,
    "saved_frames": [...],
    "saved_frames_count": 15
  },
  "unique_person": 25,
  "unique_helmet": 20,
  "unique_no_helmet": 5
}
```

## 🔧 Configuration Options

### Model Settings

- `MODEL_PATH`: Path to YOLO model weights file
- `CONF_DEFAULT`: Default confidence threshold for detections
- `LOG_LEVEL`: Logging level (debug, info, warning, error)

### Google Drive Settings

- `GDRIVE_ENABLED`: Enable/disable Google Drive integration
- `GDRIVE_CREDENTIALS_PATH`: Path to OAuth credentials file
- `GDRIVE_TOKEN_PATH`: Path to cached OAuth token
- `GDRIVE_PARENT_FOLDER_ID`: Parent folder ID for uploads
- `GDRIVE_AUTH_PORT`: Port for OAuth local server

### Frame Processing

- `MAX_SAVED_FRAMES`: Maximum number of frames to save
- `FRAME_STRIDE`: Frame sampling interval (process every Nth frame)
- `PREVIEW_ONLY_ONE`: Save only the best frame (1) or all qualifying frames (0)
- `PREVIEW_STRATEGY`: Frame selection strategy

### Frame Saving Policy

- `SAVE_ONLY_MIXED`: Save only frames with both helmeted and unhelmeted people
- `SAVE_FRAMES_WITH_HELMET`: Save frames containing people with helmets
- `SAVE_FRAMES_WITHOUT_HELMET`: Save frames containing people without helmets

## 🎯 Usage Examples

### Python Client Example

```python
import requests

# Upload and analyze a video
with open('video.mp4', 'rb') as f:
    files = {'file': f}
    data = {
        'conf': 0.4,
        'frame_stride': 10,
        'max_frames': 100,
        'track': 1
    }
    response = requests.post(
        'http://localhost:8000/analyze/helmet-detection',
        files=files,
        data=data
    )
    result = response.json()
    print(f"Detected {result['total_person']} people")
    print(f"With helmets: {result['count_helmet']}")
    print(f"Without helmets: {result['count_no_helmet']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/analyze/helmet-detection" \
  -F "file=@video.mp4" \
  -F "conf=0.4" \
  -F "frame_stride=10" \
  -F "track=1"
```

## 🔍 Detection Classes

The model detects two classes:

- **Class 0**: Person with helmet
- **Class 1**: Person without helmet

## 📊 Tracking Mode

When tracking is enabled (`track=1`), the system:

- Assigns unique IDs to detected people
- Tracks them across frames
- Provides unique person counts
- Uses configurable counting rules

### Counting Rules

- `any-no-helmet`: Count person as "no helmet" if detected without helmet in any frame
- `majority`: Count person based on majority classification across all frames

## 🚨 Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid file format or parameters
- **503 Service Unavailable**: Model not loaded or Google Drive unavailable
- **500 Internal Server Error**: Processing errors

## 🔒 Security

- OAuth 2.0 authentication for Google Drive
- CORS configuration for web security
- File type validation
- Temporary file cleanup

## 📈 Performance

- Optimized frame processing with configurable stride
- Memory-efficient video processing
- Automatic cleanup of temporary files
- Configurable output retention

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Google Drive API](https://developers.google.com/drive) for cloud storage integration

## 📞 Support

For issues and questions:

- Create an issue on GitHub
- Check the API documentation at `http://localhost:8000/docs`
- Review the health endpoint for system status

---

**Note**: Make sure to place your trained YOLO model weights in the `weights/` directory and configure Google Drive credentials before running the application.
