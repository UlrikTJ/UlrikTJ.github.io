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
from flask_cors import CORS
from OpticSimProj.Workspace.GodFunction import simulate_optical_structure

# Simple time-based cache
cache = {}
CACHE_DURATION = 60  # seconds

def create_app():
    # Create the Flask app
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
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
        """API endpoint for optical simulation"""
        data = request.json
        
        # Extract structure type
        structure_type = data.get('type_of_structure', 'taper')
        
        # Extract common parameters
        outer_radius = data.get('outerRadius', 10e-6)
        inner_radius = data.get('innerRadius', 1000e-9)
        n1 = data.get('n1', 3.5)
        n2 = data.get('n2', 1.0)
        glass_index = data.get('glassIndex', 1.49)
        modes = data.get('modes', 100)
        glass_distance = data.get('glassDistance', 1e-9)
        wavelength = data.get('wavelength', 950e-9)
        
        # Extract structure-specific parameters
        taper_angle = data.get('taperAngle', 5.0) if structure_type == 'taper' else None
        n_ar = data.get('n_ar', 1.9) if structure_type == 'ar_coating' else None
        ar_thickness = data.get('thickness_of_ar_coating', 100e-9) if structure_type == 'ar_coating' else None
        
        try:
            result = simulate_optical_structure(
                type_of_structure=structure_type,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                n1=n1,
                n2=n2,
                taper_angle=taper_angle,
                n_ar=n_ar,
                thickness_of_ar_coating=ar_thickness,
                number_of_modes=modes,
                glass_distance=glass_distance,
                glass_index=glass_index,
                wavelength=wavelength,
                return_image=True
            )
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/intensity_profile', methods=['POST'])
    def intensity_profile():
        """API endpoint for generating intensity profile data"""
        data = request.json
        
        # Extract parameters
        structure_type = data.get('type_of_structure', 'taper')  
        outer_radius = data.get('outerRadius', 10e-6)
        inner_radius = data.get('innerRadius', 1000e-9) 
        n1 = data.get('n1', 3.5)
        n2 = data.get('n2', 1.0)
        taper_angle = data.get('taperAngle', 5.0) if structure_type == 'taper' else None
        n_ar = data.get('n_ar', 1.9) if structure_type == 'ar_coating' else None
        ar_thickness = data.get('thickness_of_ar_coating', 100e-9) if structure_type == 'ar_coating' else None
        modes = data.get('modes', 100)
        glass_distance = data.get('glassDistance', 1e-9)
        wavelength = data.get('wavelength', 950e-9)
        
        try:
            # Use the new function from GodFunction.py
            from OpticSimProj.Workspace.GodFunction import get_intensity_profile
            
            result = get_intensity_profile(
                type_of_structure=structure_type,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                n1=n1,
                n2=n2,
                taper_angle=taper_angle,
                n_ar=n_ar, 
                thickness_of_ar_coating=ar_thickness,
                number_of_modes=modes,
                glass_distance=glass_distance,
                wavelength=wavelength
            )
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    def calculate_intensity_profile(distance, inner_radius, outer_radius, n1, taper_angle, modes):
        """
        Calculate intensity profile data.
        This is a placeholder - in your production code, you should use your physics models.
        """
        # In reality, you would extract this from the simulation results
        # Or call existing Python functions that calculate this properly
        return [float(np.exp(-np.power((d/(outer_radius*1e6) - 1), 2) * 5) * 
                    (1 + (n1-1)/5) * 
                    (1 - 0.3 * np.exp(-taper_angle/10) * np.sin(d/(outer_radius*1e6) * 8)))
                for d in distance]

    return app
    
if __name__ == "__main__":
    # Build and run the Flask app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)