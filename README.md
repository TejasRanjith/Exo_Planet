# ExoSeek - Flask Web Application

A beautiful space-themed web application for exploring exoplanets, built with Flask.

## Features

- 🌌 Interactive solar system animation
- 📊 Real-time exoplanet statistics
- 🔍 Advanced exoplanet analysis tool
- 🎨 Beautiful space-themed UI with glassmorphism effects
- 📱 Responsive design
- ⚡ Real-time analysis with Flask API

## Installation

1. **Clone or download the project files**

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
exoseek/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
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
- `POST /api/analyze_exoplanet` - Analyze exoplanet data

## Exoplanet Analysis

The application provides intelligent analysis of exoplanet characteristics:

- **Orbital Period Analysis**: Determines temperature zones
- **Size Analysis**: Identifies planet type (rocky, gas giant, etc.)
- **Transit Duration Analysis**: Provides insights about orbit and size
- **Habitability Score**: Calculates potential habitability (0-100)

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
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

For production deployment, consider using:
- Gunicorn or uWSGI as WSGI server
- Nginx as reverse proxy
- Environment variables for configuration

## License

This project is open source and available under the MIT License.
