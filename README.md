# Sign Language Translator - Deployment Package

## Quick Start

### 1. Install Dependencies
```bash
# Frontend
cd frontend
npm install

# Backend
cd ../backend
npm install
pip install tensorflow opencv-python mediapipe numpy pillow
```

### 2. Setup Database
- Start XAMPP MySQL
- Open phpMyAdmin
- Run `database/setup_database.sql`

### 3. Start Servers
```bash
# Terminal 1 - Backend
cd backend
node server.js

# Terminal 2 - Frontend
cd frontend
npm start
```

### 4. Access Application
- Frontend: http://localhost:3000
- Backend: http://localhost:5000

## Features
- Image to Text (ASL Alphabet A-Z)
- Webcam to Text (LSTM Video Recognition)
- Voice to Text (Speech Recognition)
- User Authentication
- Admin Dashboard
- Activity Tracking

## Models
- `models/sign_language_model.h5` - Image recognition (64x64, 26 classes)
- `models/asl_model_lstm.h5` - Video recognition (30 frames, 50 classes)

## Requirements
- Node.js 14+
- Python 3.8+
- MySQL (via XAMPP)
- Modern browser (Chrome/Edge recommended)

## Support
For issues or questions, check the documentation files.
