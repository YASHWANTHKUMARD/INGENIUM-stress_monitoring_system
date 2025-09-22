# Stress Monitoring System with AI/ML

A comprehensive stress monitoring system that uses machine learning to detect stress levels and provides personalized recommendations through GenAI integration.

## Features

- **ML-based Stress Detection**: Uses multiple algorithms to analyze stress indicators
- **Personalized Recommendations**: GenAI-powered suggestions for books, music, activities
- **Real-time Monitoring**: Continuous stress level tracking
- **Data Visualization**: Interactive dashboards and charts
- **RESTful API**: Easy integration with other systems

## Workflow

1. **Data Collection**: Gather stress indicators (heart rate, sleep, activity, mood)
2. **ML Training**: Train models on stress detection datasets
3. **Stress Detection**: Real-time analysis of user data
4. **GenAI Integration**: Generate personalized recommendations
5. **User Interface**: Display results and recommendations

## Project Structure

```
stress-monitoring-system/
├── data/                   # Datasets and sample data
├── models/                 # ML model files and training scripts
├── api/                    # FastAPI backend
├── frontend/               # Web interface
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── docs/                   # Documentation
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
python -m uvicorn api.main:app --reload
```

4. Access the web interface at `http://localhost:8000`

## API Endpoints

- `POST /api/stress/predict` - Predict stress level
- `GET /api/recommendations` - Get personalized recommendations
- `POST /api/data/upload` - Upload stress data
- `GET /api/dashboard` - Get dashboard data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
