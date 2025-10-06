#!/usr/bin/env python3
"""
Debug script to test model predictions with known data
"""

import json
import joblib
import pandas as pd
import numpy as np
import os

def test_kepler_prediction():
    """Test Kepler model with known data"""
    print("=== Testing Kepler Model ===")
    
    # Load test data
    with open('test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Load models
    models_dir = 'models'
    try:
        model = joblib.load(os.path.join(models_dir, 'ensemble_model.pkl'))
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return
    
    # Test with Kepler-452b data
    test_case = test_data['kepler_test_cases'][0]  # Earth-like planet
    print(f"\nTesting: {test_case['name']}")
    print(f"Expected: {test_case['expected']}")
    
    data = test_case['data']
    
    # Create feature mapping (same as in app.py)
    feature_mapping = {
        'koi_period': data.get('orbital_period', 100.0),
        'koi_period_err1': 1.0,
        'koi_period_err2': 1.0,
        'koi_time0bk': 2000.0,
        'koi_time0bk_err1': 0.1,
        'koi_time0bk_err2': 0.1,
        'koi_impact': 0.5,
        'koi_impact_err1': 0.1,
        'koi_impact_err2': 0.1,
        'koi_duration': data.get('transit_duration', 5.0),
        'koi_duration_err1': 0.1,
        'koi_duration_err2': 0.1,
        'koi_depth': data.get('transit_depth', 0.001),
        'koi_depth_err1': 0.0001,
        'koi_depth_err2': 0.0001,
        'koi_prad': data.get('planet_radius', 1.0),
        'koi_prad_err1': 0.1,
        'koi_prad_err2': 0.1,
        'koi_teq': data.get('equilibrium_temp', 300.0),
        'koi_teq_err1': 10.0,
        'koi_teq_err2': 10.0,
        'koi_insol': data.get('insolation_flux', 1.0),
        'koi_insol_err1': 0.1,
        'koi_insol_err2': 0.1,
        'koi_model_snr': 10.0,
        'koi_tce_plnt_num': 1,
        'koi_steff': data.get('stellar_temp', 5000.0),
        'koi_steff_err1': 100.0,
        'koi_steff_err2': 100.0,
        'koi_slogg': data.get('stellar_gravity', 4.5),
        'koi_slogg_err1': 0.1,
        'koi_slogg_err2': 0.1,
        'koi_srad': data.get('stellar_radius', 1.0),
        'koi_srad_err1': 0.1,
        'koi_srad_err2': 0.1,
        'ra': 180.0,
        'dec': 0.0,
        'koi_kepmag': data.get('kepler_magnitude', 12.0),
        'koi_score': 0.5,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0,
        'koi_pdisposition': 'CANDIDATE',
        'koi_tce_delivname': 'q1_q17_dr25_tce'
    }
    
    # Create input dictionary with original feature names
    original_features = preprocessor.feature_names_in_
    input_dict = {}
    
    for feature in original_features:
        if feature in feature_mapping:
            input_dict[feature] = feature_mapping[feature]
        else:
            # Set default values for missing features
            input_dict[feature] = 0.0
    
    print(f"\nInput features (first 10):")
    for i, (key, value) in enumerate(list(input_dict.items())[:10]):
        print(f"  {key}: {value}")
    
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Preprocess
        X_preprocessed = preprocessor.transform(input_df)
        X_selected = selector.transform(X_preprocessed)
        
        print(f"\nPreprocessed shape: {X_preprocessed.shape}")
        print(f"Selected features shape: {X_selected.shape}")
        
        # Predict
        prediction = model.predict(X_selected)[0]
        probabilities = model.predict_proba(X_selected)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"\nPrediction: {predicted_label}")
        print(f"Confidence: {max(probabilities):.3f}")
        print(f"All probabilities:")
        for cls, prob in zip(label_encoder.classes_, probabilities):
            print(f"  {cls}: {prob:.3f}")
        
        # Check if prediction matches expected
        if predicted_label == test_case['expected']:
            print(f"\n[OK] SUCCESS: Prediction matches expected result!")
        else:
            print(f"\n[ERROR] MISMATCH: Expected {test_case['expected']}, got {predicted_label}")
            
    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def test_simple_prediction():
    """Test with very simple, obvious data"""
    print("\n=== Testing with Simple Data ===")
    
    # Load models
    models_dir = 'models'
    try:
        model = joblib.load(os.path.join(models_dir, 'ensemble_model.pkl'))
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        selector = joblib.load(os.path.join(models_dir, 'selector.pkl'))
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return
    
    # Create very obvious CONFIRMED planet data
    simple_data = {
        'orbital_period': 365.0,  # Earth-like period
        'transit_duration': 13.0,  # Earth-like duration
        'transit_depth': 0.0008,   # Earth-like depth
        'planet_radius': 1.0,      # Earth radius
        'equilibrium_temp': 288,   # Earth temperature
        'insolation_flux': 1.0,    # Earth insolation
        'stellar_temp': 5778,      # Sun temperature
        'stellar_radius': 1.0,     # Sun radius
        'stellar_gravity': 4.44,   # Sun gravity
        'kepler_magnitude': 12.0   # Typical magnitude
    }
    
    print(f"Testing with Earth-like planet data:")
    for key, value in simple_data.items():
        print(f"  {key}: {value}")
    
    # Use the same feature mapping logic
    feature_mapping = {
        'koi_period': simple_data['orbital_period'],
        'koi_period_err1': 1.0,
        'koi_period_err2': 1.0,
        'koi_time0bk': 2000.0,
        'koi_time0bk_err1': 0.1,
        'koi_time0bk_err2': 0.1,
        'koi_impact': 0.5,
        'koi_impact_err1': 0.1,
        'koi_impact_err2': 0.1,
        'koi_duration': simple_data['transit_duration'],
        'koi_duration_err1': 0.1,
        'koi_duration_err2': 0.1,
        'koi_depth': simple_data['transit_depth'],
        'koi_depth_err1': 0.0001,
        'koi_depth_err2': 0.0001,
        'koi_prad': simple_data['planet_radius'],
        'koi_prad_err1': 0.1,
        'koi_prad_err2': 0.1,
        'koi_teq': simple_data['equilibrium_temp'],
        'koi_teq_err1': 10.0,
        'koi_teq_err2': 10.0,
        'koi_insol': simple_data['insolation_flux'],
        'koi_insol_err1': 0.1,
        'koi_insol_err2': 0.1,
        'koi_model_snr': 10.0,
        'koi_tce_plnt_num': 1,
        'koi_steff': simple_data['stellar_temp'],
        'koi_steff_err1': 100.0,
        'koi_steff_err2': 100.0,
        'koi_slogg': simple_data['stellar_gravity'],
        'koi_slogg_err1': 0.1,
        'koi_slogg_err2': 0.1,
        'koi_srad': simple_data['stellar_radius'],
        'koi_srad_err1': 0.1,
        'koi_srad_err2': 0.1,
        'ra': 180.0,
        'dec': 0.0,
        'koi_kepmag': simple_data['kepler_magnitude'],
        'koi_score': 0.5,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0,
        'koi_pdisposition': 'CANDIDATE',
        'koi_tce_delivname': 'q1_q17_dr25_tce'
    }
    
    # Create input dictionary
    original_features = preprocessor.feature_names_in_
    input_dict = {}
    
    for feature in original_features:
        if feature in feature_mapping:
            input_dict[feature] = feature_mapping[feature]
        else:
            input_dict[feature] = 0.0
    
    input_df = pd.DataFrame([input_dict])
    
    try:
        # Preprocess and predict
        X_preprocessed = preprocessor.transform(input_df)
        X_selected = selector.transform(X_preprocessed)
        
        prediction = model.predict(X_selected)[0]
        probabilities = model.predict_proba(X_selected)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"\nPrediction: {predicted_label}")
        print(f"Confidence: {max(probabilities):.3f}")
        print(f"All probabilities:")
        for cls, prob in zip(label_encoder.classes_, probabilities):
            print(f"  {cls}: {prob:.3f}")
            
    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kepler_prediction()
    test_simple_prediction()
