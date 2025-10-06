#!/usr/bin/env python3
"""
Test the false positive case that should be FALSE POSITIVE but is showing CONFIRMED
"""

from app import predict_kepler

def test_false_positive_case():
    """Test the false positive case from test_data.json"""
    print("=== Testing False Positive Case ===")
    
    # False positive data from test_data.json
    false_positive_data = {
        'orbital_period': 15.2,
        'transit_duration': 8.0,
        'transit_depth': 0.0001,
        'planet_radius': 0.8,
        'equilibrium_temp': 400,
        'insolation_flux': 2.5,
        'stellar_temp': 4500,
        'stellar_radius': 0.7,
        'stellar_gravity': 4.6,
        'kepler_magnitude': 14.2
    }
    
    print("Input data:")
    for key, value in false_positive_data.items():
        print(f"  {key}: {value}")
    
    print("\nChecking heuristics conditions:")
    orbital_period = false_positive_data['orbital_period']
    planet_radius = false_positive_data['planet_radius']
    equilibrium_temp = false_positive_data['equilibrium_temp']
    insolation_flux = false_positive_data['insolation_flux']
    transit_depth = false_positive_data['transit_depth']
    
    # Check Earth-like/Super-Earth conditions
    earth_like_super_earth = (50 <= orbital_period <= 1000 and 
                             0.5 <= planet_radius <= 2.5 and 
                             150 <= equilibrium_temp <= 400 and 
                             0.1 <= insolation_flux <= 5.0 and
                             transit_depth > 0.0001)
    
    # Check Hot Jupiter conditions
    hot_jupiter = (orbital_period < 20 and planet_radius > 4.0)
    
    # Check Gas giant conditions
    gas_giant = (planet_radius > 3.0 and orbital_period > 50)
    
    # Check False positive conditions
    false_positive = (planet_radius < 0.3 or 
                     equilibrium_temp > 1500 or 
                     insolation_flux > 200 or
                     transit_depth < 0.00001)
    
    print(f"Orbital period: {orbital_period} (50-1000 range: {50 <= orbital_period <= 1000})")
    print(f"Planet radius: {planet_radius} (0.5-2.5 range: {0.5 <= planet_radius <= 2.5})")
    print(f"Equilibrium temp: {equilibrium_temp} (150-400 range: {150 <= equilibrium_temp <= 400})")
    print(f"Insolation flux: {insolation_flux} (0.1-5.0 range: {0.1 <= insolation_flux <= 5.0})")
    print(f"Transit depth: {transit_depth} (>0.0001: {transit_depth > 0.0001})")
    
    print(f"\nEarth-like/Super-Earth conditions met: {earth_like_super_earth}")
    print(f"Hot Jupiter conditions met: {hot_jupiter}")
    print(f"Gas giant conditions met: {gas_giant}")
    print(f"False positive conditions met: {false_positive}")
    
    # Get prediction
    print("\nGetting prediction...")
    result = predict_kepler(false_positive_data)
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    
    # Check if this matches what we expect
    if result['prediction'] == 'FALSE POSITIVE' and result['confidence'] > 0.7:
        print("\n[SUCCESS] Prediction is correct!")
    else:
        print(f"\n[ERROR] Expected FALSE POSITIVE with >70% confidence, got {result['prediction']} with {result['confidence']:.1%}")
        print("This case should be FALSE POSITIVE because:")
        print("- Very small transit depth (0.0001) suggests stellar activity")
        print("- Small planet radius (0.8) with high temperature (400K)")
        print("- High insolation flux (2.5) for such a small planet")

if __name__ == "__main__":
    test_false_positive_case()
