#!/usr/bin/env python3
"""
Test the updated Flask app prediction logic
"""

from app import predict_kepler

def test_updated_predictions():
    """Test the updated prediction logic"""
    print("=== Testing Updated Flask App Predictions ===")
    
    # Test Earth-like planet
    earth_data = {
        'orbital_period': 365,
        'transit_duration': 13,
        'transit_depth': 0.0008,
        'planet_radius': 1.0,
        'equilibrium_temp': 288,
        'insolation_flux': 1.0,
        'stellar_temp': 5778,
        'stellar_radius': 1.0,
        'stellar_gravity': 4.44,
        'kepler_magnitude': 12.0
    }
    
    print("Testing Earth-like planet:")
    result = predict_kepler(earth_data)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")
    
    # Test Hot Jupiter
    hot_jupiter_data = {
        'orbital_period': 3.5,
        'transit_duration': 2.5,
        'transit_depth': 0.02,
        'planet_radius': 12.0,
        'equilibrium_temp': 1200,
        'insolation_flux': 1000,
        'stellar_temp': 5850,
        'stellar_radius': 1.0,
        'stellar_gravity': 4.44,
        'kepler_magnitude': 11.3
    }
    
    print("\nTesting Hot Jupiter:")
    result = predict_kepler(hot_jupiter_data)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")

if __name__ == "__main__":
    test_updated_predictions()

