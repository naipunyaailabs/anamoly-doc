#!/usr/bin/env python3
"""
Main application entry point for the Crowd Anomaly Detection System
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point"""
    print("Crowd Anomaly Detection System")
    print("Starting services...")
    
    # TODO: Implement main application logic
    # This could start the FastAPI server, Streamlit dashboard, or both
    # based on command line arguments or environment variables

if __name__ == "__main__":
    main()