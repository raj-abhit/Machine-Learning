#!/usr/bin/env python3
import subprocess
import os

os.chdir(r'c:\Users\og\Desktop\CrewAi')

# Try to push
try:
    result = subprocess.run(
        ['git', 'push', '-u', 'origin', 'main'],
        capture_output=True,
        text=True,
        timeout=30
    )
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
except subprocess.TimeoutExpired:
    print("Push timed out")
except Exception as e:
    print(f"Error: {e}")

# Check remote config
try:
    result = subprocess.run(
        ['git', 'remote', '-v'],
        capture_output=True,
        text=True
    )
    print("\n\nRemote config:")
    print(result.stdout)
except Exception as e:
    print(f"Error checking remote: {e}")
