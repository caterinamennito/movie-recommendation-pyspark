"""
Streamlit app entry point.
"""
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import and run the main app
from app import main

if __name__ == "__main__":
    main()