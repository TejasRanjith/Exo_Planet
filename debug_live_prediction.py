#!/usr/bin/env python3
"""
Debug the live prediction to see why heuristics aren't working
"""

from app import predict_kepler
import json

def debug_prediction():
    """Debug the prediction with detailed logging"""
    print("=== Debugging Live Prediction ===")
    
    # Test with Kepler-452b data (should be CONFIRMED)
    test_data = {
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
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    print("\nChecking heuristics conditions:")
    orbital_period = test_data['orbital_period']
    planet_radius = test_data['planet_radius']
    equilibrium_temp = test_data['equilibrium_temp']
    insolation_flux = test_data['insolation_flux']
    
    print(f"Orbital period: {orbital_period} (50-500 range: {50 <= orbital_period <= 500})")
    print(f"Planet radius: {planet_radius} (0.8-1.5 range: {0.8 <= planet_radius <= 1.5})")
    print(f"Equilibrium temp: {equilibrium_temp} (200-350 range: {200 <= equilibrium_temp <= 350})")
    print(f"Insolation flux: {insolation_flux} (0.5-2.0 range: {0.5 <= insolation_flux <= 2.0})")
    
    # Check new heuristics conditions
    earth_like_super_earth = (50 <= orbital_period <= 1000 and 
                             0.5 <= planet_radius <= 2.5 and 
                             150 <= equilibrium_temp <= 400 and 
                             0.1 <= insolation_flux <= 5.0 and
                             test_data['transit_depth'] > 0.0001)
    
    hot_jupiter = (orbital_period < 20 and planet_radius > 4.0)
    
    gas_giant = (planet_radius > 3.0 and orbital_period > 50)
    
    false_positive = (planet_radius < 0.3 or 
                     equilibrium_temp > 1500 or 
                     insolation_flux > 200 or
                     test_data['transit_depth'] < 0.00001)
    
    print(f"\nEarth-like/Super-Earth conditions met: {earth_like_super_earth}")
    print(f"Hot Jupiter conditions met: {hot_jupiter}")
    print(f"Gas giant conditions met: {gas_giant}")
    print(f"False positive conditions met: {false_positive}")
    
    # Get prediction
    print("\nGetting prediction...")
    result = predict_kepler(test_data)
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'probabilities' in result:
        print("Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob:.3f}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")

def test_different_scenarios():
    """Test different scenarios to understand the issue"""
    print("\n=== Testing Different Scenarios ===")
    
    scenarios = [
        {
            'name': 'Kepler-452b (should be CONFIRMED)',
            'data': {
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
        },
        {
            'name': 'Earth-like (should be CONFIRMED)',
            'data': {
                'orbital_period': 365.0,
                'transit_duration': 13.0,
                'transit_depth': 0.0008,
                'planet_radius': 1.0,
                'equilibrium_temp': 288,
                'insolation_flux': 1.0,
                'stellar_temp': 5778,
                'stellar_radius': 1.0,
                'stellar_gravity': 4.44,
                'kepler_magnitude': 12.0
            }
        },
        {
            'name': 'Hot Jupiter (should be CONFIRMED)',
            'data': {
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
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        result = predict_kepler(scenario['data'])
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        # Check which heuristics should apply
        data = scenario['data']
        orbital_period = data['orbital_period']
        planet_radius = data['planet_radius']
        equilibrium_temp = data['equilibrium_temp']
        insolation_flux = data['insolation_flux']
        
        earth_like = (50 <= orbital_period <= 500 and 
                     0.8 <= planet_radius <= 1.5 and 
                     200 <= equilibrium_temp <= 350 and 
                     0.5 <= insolation_flux <= 2.0)
        
        hot_jupiter = (orbital_period < 10 and planet_radius > 5.0)
        
        print(f"Earth-like heuristics: {earth_like}")
        print(f"Hot Jupiter heuristics: {hot_jupiter}")

if __name__ == "__main__":
    debug_prediction()
    test_different_scenarios()
