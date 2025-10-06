#!/usr/bin/env python3
"""
Test script for ExoSeek ML models
This script tests the three models (Kepler, K2, TESS) with sample data
"""

import json
import requests
import time

def test_model_prediction(model_type, test_data, base_url="http://localhost:5000"):
    """Test a single model prediction"""
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} Model")
    print(f"{'='*60}")
    
    for i, test_case in enumerate(test_data, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Expected: {test_case['expected']}")
        
        # Prepare the data
        data = test_case['data'].copy()
        data['model_type'] = model_type
        
        try:
            # Send request to Flask app
            response = requests.post(
                f"{base_url}/api/analyze_exoplanet",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    ml_prediction = result['analysis']['ml_prediction']
                    print(f"[SUCCESS] ML Prediction: {ml_prediction['prediction']}")
                    print(f"   Confidence: {ml_prediction['confidence']:.2%}")
                    
                    if 'probabilities' in ml_prediction:
                        print("   Class Probabilities:")
                        for class_name, prob in ml_prediction['probabilities'].items():
                            print(f"     {class_name}: {prob:.2%}")
                    
                    # Check if prediction matches expected
                    if ml_prediction['prediction'] == test_case['expected']:
                        print("   [MATCH] Prediction matches expected result!")
                    else:
                        print(f"   [DIFFER] Prediction differs from expected ({test_case['expected']})")
                else:
                    print(f"[ERROR] Error in prediction: {result.get('error', 'Unknown error')}")
            else:
                print(f"[ERROR] HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        
        print("-" * 40)

def main():
    """Main test function"""
    print("ExoSeek ML Models Test Suite")
    print("=" * 60)
    
    # Load test data
    try:
        with open('test_data.json', 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("❌ test_data.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing test_data.json: {e}")
        return
    
    # Test each model
    models_to_test = [
        ('kepler', 'kepler_test_cases'),
        ('k2', 'k2_test_cases'), 
        ('tess', 'tess_test_cases')
    ]
    
    for model_type, data_key in models_to_test:
        if data_key in test_data:
            test_model_prediction(model_type, test_data[data_key])
        else:
            print(f"[ERROR] No test data found for {model_type} model")
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
