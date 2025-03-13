#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
from system_info_db import init_db, produce_json_structure
import time
from functools import lru_cache

# Simple time-based cache
cache = {}
CACHE_DURATION = 60  # seconds

def create_app():
    # Create the Flask app
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)

    # Use the application context immediately after creating 'app'
    with app.app_context():
        # Initialize the DB (or call any other "before_first_request" code)
        init_db()

    @app.route("/")
    def root_endpoint():
        try:
            data = produce_json_structure()
            return jsonify(data)
        except Exception as e:
            app.logger.error(f"Error in root_endpoint: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    # In the serve_data route
    @app.route("/data.json", methods=["GET"])
    def serve_data():
        try:
            # Get query parameters for pagination
            limit = request.args.get('limit', default=100, type=int)
            # Cap the limit to prevent memory issues
            if limit > 5000:
                limit = 5000

            page = request.args.get('page', default=0, type=int)

            # Check cache first
            cache_key = f"data_{limit}_{page}"
            current_time = time.time()

            if cache_key in cache and current_time - cache[cache_key]['time'] < CACHE_DURATION:
                return jsonify(cache[cache_key]['data'])

            # Get full data structure but limit historical data
            data = produce_json_structure(historical_limit=limit, historical_offset=page*limit)

            # Store in cache
            cache[cache_key] = {
                'time': current_time,
                'data': data
            }

            return jsonify(data)
        except Exception as e:
            app.logger.error(f"Error in serve_data: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == "__main__":
    # Build and run the Flask app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)