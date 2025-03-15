#!/usr/bin/env python3
from flask import Flask, jsonify, request
from system_info_db import init_db, produce_json_structure
import time
import os
import sys
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64

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

    def generate_heatmap(inner_radius, outer_radius, n1, taper_angle, modes):
        # Create a field data matrix (replace with your actual field calculation)
        height, width = 300, 500
        field_data = np.zeros((height, width))
        
        # Your field calculation code here
        for y in range(height):
            normalized_y = (y - height/2) / (height/2)
            
            for x in range(width):
                normalized_x = x / width
                
                # Create tapered waveguide shape
                taper_width = 0.2 * (1 - normalized_x * (taper_angle/90))
                upper_boundary = taper_width
                lower_boundary = -taper_width
                
                if normalized_y <= upper_boundary and normalized_y >= lower_boundary:
                    # Core region
                    dist_from_center = abs(normalized_y) / taper_width
                    mode_patterns = (1 + np.cos(np.pi * 5 * (modes/100) * normalized_y)) / 2
                    radius_effect = np.exp(-np.power(dist_from_center * 10 * (inner_radius * 1e9 / 1000), 2))
                    
                    field_intensity = (0.6 + 0.4 * radius_effect) * mode_patterns
                    field_intensity *= 0.8 + 0.2 * np.cos(normalized_x * np.pi * 10)
                else:
                    # Cladding region
                    dist_from_boundary = min(
                        abs(normalized_y - upper_boundary),
                        abs(normalized_y - lower_boundary)
                    )
                    field_intensity = 0.5 * np.exp(-dist_from_boundary * 10)
                
                # Apply index effect
                field_intensity *= 0.5 + 0.5 * (n1 / 5)
                field_data[y, x] = min(1, max(0, field_intensity))
        
        # Set up the plot with proper units
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot the field data
        im = ax.imshow(field_data, cmap='viridis', origin='lower', aspect='auto')
        
        # Add colorbar with label
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Field Intensity (a.u.)')
        
        # Add x and y axis labels with units
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Position (μm)')
        
        # Calculate actual dimensions
        x_microns = outer_radius * 2 * 1e6  # Convert to microns
        y_microns = outer_radius * 2 * 1e6
        
        # Set ticks with actual dimensions
        x_ticks = np.linspace(0, field_data.shape[1], 5)
        y_ticks = np.linspace(0, field_data.shape[0], 5)
        x_tick_labels = [f'{x:.1f}' for x in np.linspace(0, x_microns, 5)]
        y_tick_labels = [f'{y:.1f}' for y in np.linspace(-y_microns/2, y_microns/2, 5)]
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)
        
        # Add title with parameters
        ax.set_title(f'Field Distribution (n1={n1:.2f}, Taper={taper_angle}°)')
        
        # Add annotations for waveguide
        plt.text(0.05, 0.95, 'Waveguide', transform=ax.transAxes, color='white', fontweight='bold')
        plt.text(0.7, 0.05, 'Direction →', transform=ax.transAxes, color='white')
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return f'data:image/png;base64,{img_str}'

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
            
            app.logger.info(f"Starting simulation with: R={outer_radius}, r={inner_radius}, n1={n1}, angle={taper_angle}, modes={modes}")
            
            # Import necessary functions from OpticSimProj
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), 'OpticSimProj/Workspace'))
                app.logger.info(f"Module path: {os.path.join(os.path.dirname(__file__), 'OpticSimProj/Workspace')}")
                
                from commands import xzgraph, getk_n_ml, getn_ml, getbetaclists, getpropagationmatrices
                from commands import get_T_R_list, getSTSR1_Q, ABlist
                app.logger.info("Commands module imported successfully")
            except Exception as e:
                app.logger.error(f"Error importing modules: {str(e)}")
                return jsonify({"error": f"Module import error: {str(e)}"}), 500
                
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-GUI backend
                import matplotlib.pyplot as plt
                from io import BytesIO
                import base64
                app.logger.info("Matplotlib imported successfully")
            except Exception as e:
                app.logger.error(f"Error importing matplotlib: {str(e)}")
                return jsonify({"error": f"Matplotlib import error: {str(e)}"}), 500
            
            # Setup parameters similar to your Python code
            wavelength = 950e-9
            k = 2 * np.pi / wavelength
            R = outer_radius
            size = min(modes, 150)  # Limit size to avoid memory issues
            rWG = inner_radius
            
            # Calculate alpha and other parameters
            alpha = taper_angle * np.pi / 180
            h1 = rWG  # Height before taper
            h2 = rWG / np.tan(alpha)
            
            # Create lists and arrays
            try:
                glassinfo = [1.4949, 1.4533, 1.8e-6, 1e-9]  # n1, n2, glasssize, glass distance
                Rvalues = [rWG, 0, glassinfo[2]]
                hlist = [0, h1, h1 + glassinfo[3], 2 * h1 + glassinfo[3]]
                n1list = [n1, 1, glassinfo[0]]
                n2list = [1, 1, glassinfo[1]]
                
                # Create starting vector
                start = np.zeros(size)
                start[0] = 1
                app.logger.info("Parameters set up successfully")
            except Exception as e:
                app.logger.error(f"Error setting up parameters: {str(e)}")
                return jsonify({"error": f"Parameter setup error: {str(e)}"}), 500
            
            # Run simulation calculations with proper error handling
            try:
                app.logger.info("Getting k_ml values...")
                k_ml = getk_n_ml(size, R)
                
                app.logger.info("Getting n_lm values...")
                n_lm = getn_ml(size, k_ml, R)
                
                app.logger.info("Getting betaclists...")
                bclist = getbetaclists(0, n1list, n2list, Rvalues, R, k_ml, n_lm, size, k)
                
                app.logger.info("Getting propagation matrices...")
                Plist = getpropagationmatrices(bclist, hlist)
                
                app.logger.info("Getting T_R lists...")
                forT, forR, backT, backR = get_T_R_list(bclist)
                
                app.logger.info("Getting STSR values...")
                STlist, SRlist, STrevlist, SRrevlist = getSTSR1_Q(Plist, forT, forR, backT, backR)
            except Exception as e:
                app.logger.error(f"Error in simulation calculations: {str(e)}")
                return jsonify({"error": f"Simulation calculation error: {str(e)}"}), 500
            
            # Calculate efficiency
            try:
                app.logger.info("Calculating efficiency...")
                a_q = STlist[-1] @ Plist[0] @ start
                t2 = abs(a_q[0]) ** 2
            except Exception as e:
                app.logger.error(f"Error calculating efficiency: {str(e)}")
                return jsonify({"error": f"Efficiency calculation error: {str(e)}"}), 500
            
            # Generate visualization with error handling
            try:
                app.logger.info("Generating visualization...")
                STlist2, SRlist2, STrevlist2, SRrevlist2 = getSTSR1_Q(
                    Plist[::-1], backT[::-1], backR[::-1], forT[::-1], forR[::-1]
                )
                alist, blist = ABlist(Plist, SRlist, STlist, SRrevlist2[::-1], start, SRrevlist)
                Egraphreal = np.real(xzgraph(1000, alist, blist, 0, bclist, k_ml, R, hlist, size, R))
            except Exception as e:
                app.logger.error(f"Error in visualization generation: {str(e)}")
                return jsonify({"error": f"Visualization error: {str(e)}"}), 500
            
            # Create plot with error handling
            try:
                app.logger.info("Creating plot...")
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
            except Exception as e:
                app.logger.error(f"Error creating or saving plot: {str(e)}")
                return jsonify({"error": f"Plot generation error: {str(e)}"}), 500
            
            # Calculate additional factors
            overlap_factor = np.exp(-2 * ((inner_radius/outer_radius) ** 2))
            mode_match_factor = 1 - np.exp(-modes/50)
            taper_factor = np.exp(-taper_angle/20)
            index_factor = np.sqrt(n1)/1.5
            
            app.logger.info("Simulation completed successfully")
            
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
            app.logger.error(f"Uncaught error in simulation: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app
    
if __name__ == "__main__":
    # Build and run the Flask app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)