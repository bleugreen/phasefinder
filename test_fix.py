#!/usr/bin/env python3
"""
Simple test script to verify the RuntimeError fix for state_dict loading.
This reproduces the issue reported by the user.
"""

try:
    from phasefinder import Phasefinder
    
    print("Testing Phasefinder initialization...")
    pf = Phasefinder(quiet=True)
    print("✓ Success! Phasefinder initialized without RuntimeError")
    print(f"✓ Model loaded on device: {pf.device}")
    
except RuntimeError as e:
    if "Missing key(s) in state_dict" in str(e):
        print("✗ FAILED: RuntimeError still occurring")
        print(f"Error: {e}")
        exit(1)
    else:
        print(f"✗ FAILED: Different RuntimeError: {e}")
        exit(1)
except Exception as e:
    print(f"✗ FAILED: Unexpected error: {type(e).__name__}: {e}")
    exit(1)

print("\nAll tests passed!")
