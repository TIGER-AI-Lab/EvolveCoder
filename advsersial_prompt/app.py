#!/usr/bin/env python3
"""
🎯 Main Application Entry Point

This is the new main entry point for the Adversarial Generation System.
It launches the integrated Gradio interface that allows users to:
1. Configure pipeline parameters (rounds, model, etc.)
2. Run the pipeline with real-time feedback
3. View results interactively

Usage:
    python app.py
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the environment and check dependencies."""
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("💡 Please set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Check for required dependencies
    try:
        import gradio
        import plotly
        import pandas
        import numpy
        print("✅ All dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install gradio plotly pandas numpy")
        return False
    
    return True

def main():
    """Main application entry point."""
    
    print("🚀 Starting Integrated Adversarial Generation System...")
    print("=" * 60)
    
    if not setup_environment():
        return 1
    
    try:
        from integrated_gradio_app import create_integrated_interface
        
        print("🎨 Creating integrated interface...")
        app = create_integrated_interface()
        
        print("🌐 Launching web application...")
        print("📱 Access the application at: http://localhost:7860")
        print("🔗 A shareable link will also be generated")
        print("=" * 60)
        print("💡 Features available:")
        print("   🎯 Configure pipeline parameters")
        print("   🚀 Run adversarial generation with real-time feedback")
        print("   📊 View interactive visualizations")
        print("   📋 Monitor execution logs")
        print("=" * 60)
        
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            debug=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
