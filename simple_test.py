#!/usr/bin/env python3
"""
Simple test for ExoSeek Flask app without ML models
This tests the basic functionality and form handling
"""

import requests
import json

def test_basic_functionality():
    """Test basic Flask app functionality"""
    print("Testing ExoSeek Flask App Basic Functionality")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test 1: Home page
    print("\n1. Testing Home Page...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   [SUCCESS] Home page loads correctly")
        else:
            print(f"   [ERROR] Home page failed: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Home page request failed: {e}")
    
    # Test 2: Find Exoplanets page
    print("\n2. Testing Find Exoplanets Page...")
    try:
        response = requests.get(f"{base_url}/find_exoplanets")
        if response.status_code == 200:
            print("   [SUCCESS] Find Exoplanets page loads correctly")
        else:
            print(f"   [ERROR] Find Exoplanets page failed: {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] Find Exoplanets page request failed: {e}")
    
    # Test 3: API endpoint with basic data
    print("\n3. Testing API Endpoint...")
    test_data = {
        "model_type": "kepler",
        "orbital_period": 365.0,
        "transit_duration": 10.0,
        "transit_depth": 0.001,
        "planet_radius": 1.0,
        "equilibrium_temp": 300.0,
        "insolation_flux": 1.0,
        "stellar_temp": 5778.0,
        "stellar_radius": 1.0,
        "stellar_gravity": 4.44,
        "kepler_magnitude": 12.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/analyze_exoplanet",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("   [SUCCESS] API endpoint responds correctly")
            print(f"   Success: {result.get('success', False)}")
            
            if 'analysis' in result:
                analysis = result['analysis']
                print("   Analysis includes:")
                for key in analysis.keys():
                    print(f"     - {key}")
                
                if 'ml_prediction' in analysis:
                    ml_pred = analysis['ml_prediction']
                    print(f"   ML Prediction: {ml_pred.get('prediction', 'N/A')}")
                    print(f"   Confidence: {ml_pred.get('confidence', 0):.2%}")
                else:
                    print("   [NOTE] No ML prediction (models may not be loaded)")
        else:
            print(f"   [ERROR] API endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   [ERROR] API request failed: {e}")
    
    print("\n" + "=" * 50)
    print("Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()
