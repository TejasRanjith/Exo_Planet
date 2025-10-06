#!/usr/bin/env python3
"""
Test Kepler-452b prediction specifically
"""

from app import predict_kepler

def test_kepler_452b():
    """Test Kepler-452b prediction with exact data from the image"""
    print("=== Testing Kepler-452b Prediction ===")
    
    # Exact data that should result in CONFIRMED
    kepler_452b_data = {
        'orbital_period': 384.8,
        'transit_duration': 12.5,
        'transit_depth': 0.0008,
        'planet_radius': 1.63,
        'equilibrium_temp': 265,
        'insolation_flux': 1.1,
        'stellar_temp': 5753,
        'stellar_radius': 1.11,
        'stellar_gravity': 4.32,
        'kepler_magnitude': 13.4
    }
    
    print("Input data:")
    for key, value in kepler_452b_data.items():
        print(f"  {key}: {value}")
    
    print("\nChecking heuristics conditions:")
    orbital_period = kepler_452b_data['orbital_period']
    planet_radius = kepler_452b_data['planet_radius']
    equilibrium_temp = kepler_452b_data['equilibrium_temp']
    insolation_flux = kepler_452b_data['insolation_flux']
    transit_depth = kepler_452b_data['transit_depth']
    
    # Check Earth-like/Super-Earth conditions
    earth_like_super_earth = (50 <= orbital_period <= 1000 and 
                             0.5 <= planet_radius <= 2.5 and 
                             150 <= equilibrium_temp <= 400 and 
                             0.1 <= insolation_flux <= 5.0 and
                             transit_depth > 0.0001)
    
    print(f"Orbital period: {orbital_period} (50-1000 range: {50 <= orbital_period <= 1000})")
    print(f"Planet radius: {planet_radius} (0.5-2.5 range: {0.5 <= planet_radius <= 2.5})")
    print(f"Equilibrium temp: {equilibrium_temp} (150-400 range: {150 <= equilibrium_temp <= 400})")
    print(f"Insolation flux: {insolation_flux} (0.1-5.0 range: {0.1 <= insolation_flux <= 5.0})")
    print(f"Transit depth: {transit_depth} (>0.0001: {transit_depth > 0.0001})")
    print(f"Earth-like/Super-Earth conditions met: {earth_like_super_earth}")
    
    # Get prediction
    print("\nGetting prediction...")
    result = predict_kepler(kepler_452b_data)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    
    # Check if this matches what we expect
    if result['prediction'] == 'CONFIRMED' and result['confidence'] > 0.8:
        print("\n[SUCCESS] Prediction is correct!")
    else:
        print(f"\n[ERROR] Expected CONFIRMED with >80% confidence, got {result['prediction']} with {result['confidence']:.1%}")

if __name__ == "__main__":
    test_kepler_452b()
