# Facial Expression Recognition for HR

AI-powered facial expression analysis system for recruitment processes. Analyzes candidates' emotions during video interviews to provide insights for HR professionals.

📹 [Watch Project Explanation on YouTube](https://youtu.be/1HvwPApSk9E)

## Features

- 📹 **Video-based Analysis** - Upload candidate interview videos for frame-by-frame emotion detection
- 🎯 **7 Emotion Categories** - Detects angry, disgust, fear, happy, sad, surprise, neutral
- 📊 **Temporal Analysis** - Emotion timeline throughout the interview
- 🏆 **Top 3 Ranking** - Highlights dominant emotions with medal system
- 🤖 **AI Insights** - Generates evaluation insights using Google Gemini
- 📈 **Visual Reports** - Charts, statistics, and exportable JSON reports
- 🔌 **REST API** - FastAPI endpoints for integration
- 🖥️ **Web Interface** - Streamlit-based recruiter dashboard

## Tech Stack

- **Python 3.10**
- **OpenCV** - Video processing
- **MediaPipe** - Face detection
- **DeepFace** - Emotion recognition
- **FastAPI** - REST API
- **Streamlit** - Web interface
- **Google Gemini** - AI insights
- **Docker** - Containerization

## Project Structure

```
facial-analyzer/
├── docker-compose.yml       # Container orchestration
├── Dockerfile              # Container image
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore
├── README.md
└──  src/                   # Source code
    ├── __init__.py
    ├── analyzer.py        # Core facial analysis logic
    ├── api.py            # FastAPI REST endpoints
    └── app.py            # Streamlit recruiter interface
```

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd projeto-facial-rh
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get your free Gemini API key at: https://makersuite.google.com/app/apikey

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Access Services
- **Streamlit Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## How It Works

### For HR Recruiters:

1. **Upload Videos** - Upload candidate's video responses to interview questions
2. **Automatic Analysis** - System processes video frame-by-frame
3. **View Results** - See emotion timeline, top 3 emotions with medals 🥇🥈🥉
4. **AI Insights** - Generate detailed evaluation insights
5. **Export Data** - Download charts, statistics tables, and complete analysis files

### API Endpoints:

```bash
# Start analysis session
POST /session/start?question=<question_text>

# Analyze video frame
POST /analyze/frame
  - Upload: video frame image
  - Returns: emotion data

# Get session summary
GET /session/{session_id}/summary

# Generate AI insights
POST /session/{session_id}/insights

# Export full report
POST /export/report
```

## Test Dataset

Need sample videos for testing?

- **Pexels Videos**: https://www.pexels.com/search/videos/person%20talking/
- **Pixabay**: https://pixabay.com/videos/search/person%20speaking/
- Or record your own 30-second interview responses

## Configuration

### Environment Variables (.env)

```bash
GEMINI_API_KEY=your_api_key_here
API_URL=http://api:8000
```

### Docker Services

- **api** - FastAPI backend (port 8000)
- **frontend** - Streamlit interface (port 8501)

## Development

### Without Docker (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn src.api:app --reload --port 8000

# Run Streamlit (separate terminal)
streamlit run src/app.py
```

## Important Ethical Considerations

⚠️ **This system must be used responsibly:**

- ✅ Obtain explicit consent from candidates before analysis
- ✅ Inform candidates about the emotion analysis process
- ✅ Use only as a complementary evaluation tool
- ✅ Be aware of potential algorithmic biases
- ✅ Comply with GDPR and local data protection regulations
- ❌ **Never** use as the sole criterion for hiring decisions

## Limitations

- Emotion detection accuracy varies with video quality
- Cultural differences may affect expression interpretation
- System analyzes visible expressions, not internal feelings
- Best used as one of multiple evaluation factors

## License

MIT

---

**Note**: This is an educational/research project. Ensure compliance with all applicable laws and ethical guidelines before production use.
