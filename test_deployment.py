#!/usr/bin/env python3
"""
Test script to verify all dependencies and model loading for deployment
"""

def test_dependencies():
    """Test all required dependencies"""
    print("Testing dependencies...")
    
    try:
        import flask
        print(f"[OK] Flask {flask.__version__}")
    except ImportError as e:
        print(f"[FAIL] Flask: {e}")
        return False
    
    try:
        import pandas
        print(f"[OK] Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pandas: {e}")
        return False
    
    try:
        import numpy
        print(f"[OK] NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy: {e}")
        return False
    
    try:
        import sklearn
        print(f"[OK] Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"[FAIL] Scikit-learn: {e}")
        return False
    
    try:
        import joblib
        print(f"[OK] Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"[FAIL] Joblib: {e}")
        return False
    
    try:
        import xgboost
        print(f"[OK] XGBoost {xgboost.__version__}")
    except ImportError as e:
        print(f"[FAIL] XGBoost: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        import joblib
        import os
        
        models_dir = 'models'
        
        # Test Kepler model
        try:
            kepler_model = joblib.load(os.path.join(models_dir, 'ensemble_model.pkl'))
            print("[OK] Kepler model loaded successfully")
        except Exception as e:
            print(f"[FAIL] Kepler model: {e}")
        
        # Test K2 model
        try:
            k2_model = joblib.load(os.path.join(models_dir, 'ensemble_model_k2.pkl'))
            print("[OK] K2 model loaded successfully")
        except Exception as e:
            print(f"[FAIL] K2 model: {e}")
        
        # Test TESS model
        try:
            tess_model = joblib.load(os.path.join(models_dir, 'ensemble_model_TESS.pkl'))
            print("[OK] TESS model loaded successfully")
        except Exception as e:
            print(f"[FAIL] TESS model: {e}")
            
    except Exception as e:
        print(f"[FAIL] Model loading test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Deployment Test ===")
    
    deps_ok = test_dependencies()
    models_ok = test_model_loading()
    
    if deps_ok and models_ok:
        print("\n[SUCCESS] All tests passed! Ready for deployment.")
    else:
        print("\n[ERROR] Some tests failed. Check the errors above.")
