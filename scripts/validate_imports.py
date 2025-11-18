#!/usr/bin/env python3
"""Validate that all advertised functions can actually be imported."""
import sys
import os

# Silence TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_basic_import():
    """Test basic caracal import."""
    try:
        import caracal
        print(f"✓ Caracal v{caracal.__version__} imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import caracal: {e}")
        return False

def test_all_exports():
    """Test that everything in __all__ actually exists."""
    import caracal
    
    missing = []
    for name in caracal.__all__:
        if not hasattr(caracal, name):
            missing.append(name)
    
    if missing:
        print(f"✗ Missing exports: {missing}")
        return False
    else:
        print(f"✓ All {len(caracal.__all__)} exports verified")
        return True


def test_common_imports():
    """Test common import patterns."""
    import caracal

    # Required imports (must always work)
    required = [
        ("ModelConfig", "from caracal import ModelConfig"),
        ("plot_variability_summary", "from caracal import plot_variability_summary"),
        ("compare_two_models", "from caracal import compare_two_models"),
    ]

    # Optional imports (depend on optional dependencies)
    optional = [
        ("KerasModelWrapper", "from caracal import KerasModelWrapper", caracal.TENSORFLOW_AVAILABLE),
    ]

    success = 0
    total = len(required)

    # Test required imports
    for name, import_stmt in required:
        try:
            exec(import_stmt)
            print(f"✓ {name}")
            success += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")

    # Test optional imports
    for name, import_stmt, available in optional:
        if available:
            total += 1
            try:
                exec(import_stmt)
                print(f"✓ {name}")
                success += 1
            except ImportError as e:
                print(f"✗ {name}: {e}")
        else:
            print(f"⊘ {name} (skipped - optional dependency not installed)")

    return success == total

def main():
    print("=" * 60)
    print("CARACAL IMPORT VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("All Exports Exist", test_all_exports),
        ("Common Imports", test_common_imports),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
