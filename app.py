"""
Main entry point for RAG Q&A System
Run this file to start the Streamlit application
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

if __name__ == "__main__":
    # Set environment variables if .env file exists
    env_file = current_dir / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            print("python-dotenv not installed. Skipping .env file loading.")
    
    # Import and run the main app
    try:
        from ui import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install the required dependencies by running:")
        print("pip install -r requirements.txt")