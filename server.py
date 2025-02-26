#!/usr/bin/env python3
from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello! This server is providing data.json."

@app.route("/data.json", methods=["GET"])
def serve_data():
    # Try reading data.json locally
    try:
        with open("data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"error": "data.json not found."}
    except json.JSONDecodeError:
        data = {"error": "data.json is invalid JSON."}
    return jsonify(data)

if __name__ == "__main__":
    # Run on port 5000, accessible from any IP
    app.run(host="0.0.0.0", port=5000, debug=True)
