#!/usr/bin/env python3

###############################################################################
# server.py
###############################################################################
from flask import Flask, jsonify
from system_info_db import init_db, produce_json_structure

app = Flask(__name__)

@app.before_first_request
def setup_db():
    # Ensure the DB/tables exist. Called once at startup.
    init_db()

@app.route("/")
def root_endpoint():
    # Return the entire data set in JSON form
    data = produce_json_structure()
    return jsonify(data)

# Optionally keep /data.json for a direct endpoint
@app.route("/data.json", methods=["GET"])
def serve_data():
    data = produce_json_structure()
    return jsonify(data)

if __name__ == "__main__":
    # Run the Flask app on 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=True)