# ExoSeek - Flask Web Application

A beautiful space-themed web application for exploring exoplanets, built with Flask.

## Features

- 🌌 Interactive solar system animation
- 📊 Real-time exoplanet statistics
- 🔍 Advanced exoplanet analysis tool
- 🎨 Beautiful space-themed UI with glassmorphism effects
- 📱 Responsive design
- ⚡ Real-time analysis with Flask API
- 🤖 Machine Learning models for exoplanet classification

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TejasRanjith/Exo_Planet.git
   cd Exo_Planet
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Open your browser and visit:**
   ```
   http://localhost:5000
   ```

## Project Structure

```
Exo_Planet/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── render.yaml           # Render service configuration
├── models/               # Machine Learning models
│   ├── ensemble_model.pkl
│   ├── ensemble_model_k2.pkl
│   ├── ensemble_model_TESS.pkl
│   └── ... (other model files)
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   └── find_exoplanets.html  # Exoplanet finder page
└── static/               # Static files
    ├── style.css         # CSS styles
    └── script.js         # JavaScript functionality
```

## API Endpoints

- `GET /` - Home page
- `GET /find_exoplanets` - Exoplanet finder page
- `POST /api/analyze_exoplanet` - Analyze exoplanet data with ML models

## Exoplanet Analysis

The application provides intelligent analysis of exoplanet characteristics:

- **Orbital Period Analysis**: Determines temperature zones
- **Size Analysis**: Identifies planet type (rocky, gas giant, etc.)
- **Transit Duration Analysis**: Provides insights about orbit and size
- **Habitability Score**: Calculates potential habitability (0-100)
- **ML Predictions**: Uses trained models for Kepler, K2, and TESS data

## Machine Learning Models

The application includes pre-trained ensemble models for:
- **Kepler Mission Data**: Classification of exoplanet candidates
- **K2 Mission Data**: Extended Kepler mission analysis
- **TESS Mission Data**: Transiting Exoplanet Survey Satellite data

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Libraries**: scikit-learn, pandas, numpy, joblib
- **Styling**: Custom CSS with glassmorphism effects
- **Animations**: CSS animations and transitions
- **API**: RESTful API with JSON responses

## Development

To run in development mode:
```bash
export FLASK_ENV=development
python app.py
```

## Deployment

This application is ready for deployment on Render:

1. **Connect your GitHub repository to Render**
2. **Render will auto-detect the Python/Flask app**
3. **Deploy with the included Procfile and render.yaml**

The app includes:
- Gunicorn WSGI server configuration
- Production-ready requirements
- Environment variable support

## Live Demo

Deploy this app on Render for a live demo of exoplanet analysis!

## License

This project is open source and available under the MIT License.
