#!/usr/bin/env python3
from flask import Flask, jsonify, request
from system_info_db import init_db, produce_json_structure
import time
import os
import sys
import numpy as np
from functools import lru_cache

# Simple time-based cache
cache = {}
CACHE_DURATION = 60  # seconds

def create_app():
    # Create the Flask app
    app = Flask(__name__)
    
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

    @app.route("/simulate", methods=["POST"])
    def simulate():
        try:
            # Get parameters from request
            data = request.json
            outer_radius = float(data.get('outerRadius', 10e-6))
            inner_radius = float(data.get('innerRadius', 1000e-9))
            n1 = float(data.get('n1', 3.5))
            taper_angle = float(data.get('taperAngle', 5))
            modes = int(data.get('modes', 100))
            
            # Import necessary functions from OpticSimProj
            sys.path.append(os.path.join(os.path.dirname(__file__), 'OpticSimProj/Workspace'))
            from commands import xzgraph, getk_n_ml, getn_ml, getbetaclists, getpropagationmatrices
            from commands import get_T_R_list, getSTSR1_Q, ABlist
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            from io import BytesIO
            import base64
            
            # Setup parameters similar to your Python code
            wavelength = 950e-9
            k = 2 * np.pi / wavelength
            R = outer_radius
            size = modes
            rWG = inner_radius
            
            # Calculate alpha and other parameters
            alpha = taper_angle * np.pi / 180
            h1 = rWG  # Height before taper
            h2 = rWG / np.tan(alpha)
            
            # Create lists and arrays
            glassinfo = [1.4949, 1.4533, 1.8e-6, 1e-9]  # n1, n2, glasssize, glass distance
            Rvalues = [rWG, 0, glassinfo[2]]
            hlist = [0, h1, h1 + glassinfo[3], 2 * h1 + glassinfo[3]]
            n1list = [n1, 1, glassinfo[0]]
            n2list = [1, 1, glassinfo[1]]
            
            # Create starting vector
            start = np.zeros(size)
            start[0] = 1
            
            # Run simulation calculations
            k_ml = getk_n_ml(size, R)
            n_lm = getn_ml(size, k_ml, R)
            bclist = getbetaclists(0, n1list, n2list, Rvalues, R, k_ml, n_lm, size, k)
            Plist = getpropagationmatrices(bclist, hlist)
            forT, forR, backT, backR = get_T_R_list(bclist)
            STlist, SRlist, STrevlist, SRrevlist = getSTSR1_Q(Plist, forT, forR, backT, backR)
            
            # Calculate efficiency
            a_q = STlist[-1] @ Plist[0] @ start
            t2 = abs(a_q[0]) ** 2
            
            # Generate visualization
            STlist2, SRlist2, STrevlist2, SRrevlist2 = getSTSR1_Q(
                Plist[::-1], backT[::-1], backR[::-1], forT[::-1], forR[::-1]
            )
            alist, blist = ABlist(Plist, SRlist, STlist, SRrevlist2[::-1], start, SRrevlist)
            Egraphreal = np.real(xzgraph(1000, alist, blist, 0, bclist, k_ml, R, hlist, size, R))
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.matshow(Egraphreal, origin='lower', cmap='viridis')
            plt.colorbar(label='Field Intensity')
            plt.title('Electric Field Distribution')
            
            # Save plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Calculate additional factors for the frontend
            overlap_factor = np.exp(-2 * ((inner_radius/outer_radius) ** 2))
            mode_match_factor = 1 - np.exp(-modes/50)
            taper_factor = np.exp(-taper_angle/20)
            index_factor = np.sqrt(n1)/1.5
            
            # Return data and image
            return jsonify({
                'efficiency': float(t2 * 100),
                'heatmap': f'data:image/png;base64,{image_data}',
                'factors': {
                    'overlapFactor': float(overlap_factor * 100),
                    'modeMatchFactor': float(mode_match_factor * 100),
                    'taperFactor': float(taper_factor * 100),
                    'indexFactor': float(index_factor * 100)
                }
            })
            
        except Exception as e:
            app.logger.error(f"Error in simulation: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app
    
if __name__ == "__main__":
    # Build and run the Flask app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)