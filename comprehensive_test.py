#!/usr/bin/env python3
"""
Comprehensive test for ExoSeek ML models with real test data
"""

import requests
import json

def test_kepler_model():
    """Test Kepler model with real exoplanet data"""
    print("Testing Kepler Model with Real Data")
    print("=" * 50)
    
    # Test data for Kepler-452b (Earth-like planet)
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
            
            if 'analysis' in result and 'ml_prediction' in result['analysis']:
                ml_pred = result['analysis']['ml_prediction']
                print(f"ML Prediction: {ml_pred.get('prediction', 'N/A')}")
                print(f"Confidence: {ml_pred.get('confidence', 0):.2%}")
                
                if 'error' in ml_pred:
                    print(f"Error Details: {ml_pred['error']}")
                
                if 'probabilities' in ml_pred:
                    print("Class Probabilities:")
                    for class_name, prob in ml_pred['probabilities'].items():
                        print(f"  {class_name}: {prob:.2%}")
            else:
                print("No ML prediction in response")
        else:
            print(f"ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"ERROR: Request failed: {e}")

def test_k2_model():
    """Test K2 model with real exoplanet data"""
    print("\nTesting K2 Model with Real Data")
    print("=" * 50)
    
    # Test data for K2 confirmed planet
    test_data = {
        "model_type": "k2",
        "orbital_period": 4.5,
        "transit_duration": 3.2,
        "transit_depth": 0.001,
        "planet_radius": 1.2,
        "equilibrium_temp": 350,
        "insolation_flux": 1.5,
        "stellar_temp": 5200,
        "stellar_radius": 0.9,
        "stellar_gravity": 4.4,
        "planet_mass": 1.5,
        "stellar_mass": 0.95,
        "stellar_distance": 150.0,
        "disc_facility": "K2"
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
            
            if 'analysis' in result and 'ml_prediction' in result['analysis']:
                ml_pred = result['analysis']['ml_prediction']
                print(f"ML Prediction: {ml_pred.get('prediction', 'N/A')}")
                print(f"Confidence: {ml_pred.get('confidence', 0):.2%}")
                
                if 'error' in ml_pred:
                    print(f"Error Details: {ml_pred['error']}")
        else:
            print(f"ERROR: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"ERROR: Request failed: {e}")

def test_tess_model():
    """Test TESS model with real exoplanet data"""
    print("\nTesting TESS Model with Real Data")
    print("=" * 50)
    
    # Test data for TESS confirmed planet
    test_data = {
        "model_type": "tess",
        "orbital_period": 8.2,
        "transit_duration": 4.1,
        "transit_depth": 0.002,
        "planet_radius": 2.1,
        "equilibrium_temp": 420,
        "insolation_flux": 2.0,
        "stellar_temp": 5500,
        "stellar_radius": 1.05,
        "stellar_gravity": 4.35,
        "stellar_tmag": 11.5,
        "transit_midpoint": 2458000.0,
        "stellar_distance": 120.0
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
            
            if 'analysis' in result and 'ml_prediction' in result['analysis']:
                ml_pred = result['analysis']['ml_prediction']
                print(f"ML Prediction: {ml_pred.get('prediction', 'N/A')}")
                print(f"Confidence: {ml_pred.get('confidence', 0):.2%}")
                
                if 'error' in ml_pred:
                    print(f"Error Details: {ml_pred['error']}")
        else:
            print(f"ERROR: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"ERROR: Request failed: {e}")

if __name__ == "__main__":
    print("ExoSeek ML Models Comprehensive Test")
    print("=" * 60)
    
    test_kepler_model()
    test_k2_model()
    test_tess_model()
    
    print("\n" + "=" * 60)
    print("Comprehensive test completed!")
