"""WSGI entry point for the retirement planner application."""

import os
import sys
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Get port from environment variable or command line argument
    port = 5000  # default
    
    # Check for PORT environment variable (used by Render, Heroku, etc.)
    if "PORT" in os.environ:
        port = int(os.environ["PORT"])
    
    # Check for --port command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "--port" and len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # Run the application
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)
