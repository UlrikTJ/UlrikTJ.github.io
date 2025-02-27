#!/usr/bin/env python3

from flask import Flask, jsonify
from system_info_db import init_db, produce_json_structure

def create_app():
    # Create the Flask app
    app = Flask(__name__)

    # Use the application context immediately after creating 'app'
    with app.app_context():
        # Initialize the DB (or call any other "before_first_request" code)
        init_db()

    @app.route("/")
    def root_endpoint():
        data = produce_json_structure()
        return jsonify(data)

    @app.route("/data.json", methods=["GET"])
    def serve_data():
        data = produce_json_structure()
        return jsonify(data)

    return app

if __name__ == "__main__":
    # Build and run the Flask app
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)