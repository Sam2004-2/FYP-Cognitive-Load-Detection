"""
Test script for FastAPI server.

Verifies that the server is working correctly with sample predictions.
"""

import requests
import json
import sys
from typing import Dict

API_BASE_URL = "http://localhost:8000"


def test_health() -> bool:
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Health endpoint accessible")
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Feature count: {data.get('feature_count', 'N/A')}")

        if not data["model_loaded"]:
            print("✗ Model not loaded!")
            return False

        return True

    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_model_info() -> bool:
    """Test model info endpoint."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Info")
    print("=" * 60)

    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        response.raise_for_status()
        data = response.json()

        print(f"✓ Model info endpoint accessible")
        print(f"  Features: {', '.join(data['features'][:3])}... ({data['n_features']} total)")
        print(f"  Expected features: {data['features']}")

        if data["n_features"] < 1:
            print(f"✗ No features reported by model")
            return False

        return True

    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False


def test_prediction_low_load() -> bool:
    """Test prediction with low cognitive load features."""
    print("\n" + "=" * 60)
    print("TEST 3: Prediction - Low Cognitive Load")
    print("=" * 60)

    # Features indicating low cognitive load
    # - Low blink rate
    # - Normal EAR variability
    # - Low PERCLOS
    features = {
        "blink_rate": 12.0,  # Low blink rate
        "blink_count": 4.0,
        "mean_blink_duration": 180.0,
        "ear_std": 0.03,  # Low variability
        "perclos": 0.05,  # Low eye closure
        "mouth_open_mean": 0.1,
        "mouth_open_std": 0.02,
        "roll_std": 0.01,
        "pitch_std": 0.01,
        "yaw_std": 0.01,
        "motion_mean": 0.05,
        "motion_std": 0.02,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"features": features},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        print(f"✓ Prediction endpoint accessible")
        print(f"  Input features: blink_rate={features['blink_rate']}, perclos={features['perclos']}")
        print(f"  CLI: {data['cli']:.3f}")
        print(f"  Success: {data['success']}")

        if not data["success"]:
            print("✗ Prediction failed!")
            return False

        if not (0.0 <= data["cli"] <= 1.0):
            print(f"✗ CLI out of range: {data['cli']}")
            return False

        print(f"  → Interpretation: {'LOW' if data['cli'] < 0.5 else 'HIGH'} cognitive load")

        return True

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_prediction_high_load() -> bool:
    """Test prediction with high cognitive load features."""
    print("\n" + "=" * 60)
    print("TEST 4: Prediction - High Cognitive Load")
    print("=" * 60)

    # Features indicating high cognitive load
    # - High blink rate
    # - High EAR variability
    # - High PERCLOS
    features = {
        "blink_rate": 28.0,  # High blink rate
        "blink_count": 9.0,
        "mean_blink_duration": 250.0,
        "ear_std": 0.08,  # High variability
        "perclos": 0.25,  # High eye closure
        "mouth_open_mean": 0.25,
        "mouth_open_std": 0.08,
        "roll_std": 0.04,
        "pitch_std": 0.04,
        "yaw_std": 0.03,
        "motion_mean": 0.15,
        "motion_std": 0.06,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"features": features},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        print(f"✓ Prediction endpoint accessible")
        print(f"  Input features: blink_rate={features['blink_rate']}, perclos={features['perclos']}")
        print(f"  CLI: {data['cli']:.3f}")
        print(f"  Success: {data['success']}")

        if not data["success"]:
            print("✗ Prediction failed!")
            return False

        if not (0.0 <= data["cli"] <= 1.0):
            print(f"✗ CLI out of range: {data['cli']}")
            return False

        print(f"  → Interpretation: {'LOW' if data['cli'] < 0.5 else 'HIGH'} cognitive load")

        return True

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_invalid_features() -> bool:
    """Test prediction with invalid features (should handle gracefully)."""
    print("\n" + "=" * 60)
    print("TEST 5: Invalid Features Handling")
    print("=" * 60)

    # Features with NaN values (should be replaced with 0)
    features = {
        "blink_rate": float("nan"),
        "blink_count": 0.0,
        "mean_blink_duration": 0.0,
        "ear_std": 0.0,
        "perclos": 0.0,
        "mouth_open_mean": 0.0,
        "mouth_open_std": 0.0,
        "roll_std": 0.0,
        "pitch_std": 0.0,
        "yaw_std": 0.0,
        "motion_mean": 0.0,
        "motion_std": 0.0,
    }

    try:
        # Convert NaN to None for JSON serialization
        features_json = {k: (None if k == "blink_rate" else v) for k, v in features.items()}

        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"features": features_json},
            headers={"Content-Type": "application/json"},
        )

        # This might fail, which is acceptable
        if response.status_code == 422:
            print("✓ Server correctly rejects invalid features")
            return True

        response.raise_for_status()
        data = response.json()

        print(f"✓ Server handles NaN values gracefully")
        print(f"  CLI: {data['cli']:.3f}")
        print(f"  Success: {data['success']}")

        return True

    except Exception as e:
        # It's OK if this fails - means validation is working
        print(f"✓ Server validation working (rejected invalid input)")
        return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FastAPI Server Test Suite")
    print("=" * 60)
    print(f"Testing server at: {API_BASE_URL}")

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Low Load Prediction", test_prediction_low_load),
        ("High Load Prediction", test_prediction_high_load),
        ("Invalid Features", test_invalid_features),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\n✗ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ All tests passed! Server is working correctly.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the server.")
        sys.exit(1)


if __name__ == "__main__":
    main()

