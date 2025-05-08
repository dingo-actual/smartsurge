import sys
import os
from pathlib import Path


# Add multiple possible paths to ensure resolution
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root / "src"

# Add to path if not already there
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Print paths for debugging
def pytest_sessionstart(session):
    print("\nPython path during test:")
    for p in sys.path:
        print(f"  {p}")
    print(f"\nLooking for smartsurge in: {src_path}")
    print(f"Current working directory: {os.getcwd()}")