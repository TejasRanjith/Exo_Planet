#!/usr/bin/env python3
"""
Simple web interface test for ExoSeek
"""

import requests
import json

def test_web_interface():
    """Test the web interface with sample data"""
    print("Testing ExoSeek Web Interface")
    print("=" * 50)
    
    # Test data for Kepler model (which we know works)
    test_data = {
        "model_type": "kepler",
        "orbital_period": 384.8,
        "transit_duration": 12.5,
        "transit_depth": 0.0008,
        "planet_radius": 1.63,
        "equilibrium_temp": 265,
        "insolation_flux": 1.1,
        "stellar_temp": 5753,
        "stellar_radius": 1.11,
        "stellar_gravity": 4.32,
        "kepler_magnitude": 13.4
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/api/analyze_exoplanet",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: API responded correctly")
            print(f"Success: {result.get('success', False)}")
            
            if 'analysis' in result:
                analysis = result['analysis']
                print("\nAnalysis Results:")
                print(f"- Habitability Score: {analysis.get('habitability_score', 'N/A')}")
                
                if 'ml_prediction' in analysis:
                    ml_pred = analysis['ml_prediction']
                    print(f"- ML Prediction: {ml_pred.get('prediction', 'N/A')}")
                    print(f"- Confidence: {ml_pred.get('confidence', 0):.2%}")
                    
                    if 'probabilities' in ml_pred:
                        print("- Class Probabilities:")
                        for class_name, prob in ml_pred['probabilities'].items():
                            print(f"  {class_name}: {prob:.2%}")
                
                if 'orbital_period_analysis' in analysis:
                    print(f"- Orbital Period Analysis: {analysis['orbital_period_analysis'].get('description', 'N/A')}")
                
                if 'size_analysis' in analysis:
                    print(f"- Size Analysis: {analysis['size_analysis'].get('description', 'N/A')}")
                
                if 'transit_analysis' in analysis:
                    print(f"- Transit Analysis: {analysis['transit_analysis'].get('description', 'N/A')}")
        else:
            print(f"ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"ERROR: Request failed: {e}")

if __name__ == "__main__":
    test_web_interface()

