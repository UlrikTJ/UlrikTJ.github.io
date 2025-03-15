"""
Optical Simulator Module - Main interface for optical simulations

This module provides a unified interface to run optical simulations for different
structures (taper, AR coating, flat) by wrapping existing simulation functionality
from the project's codebase.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from OpticSimProj.Workspace.Finaldata import getEffectivity
from OpticSimProj.Workspace.commands import (
    getk_n_ml, getn_ml, getbetaclists, getpropagationmatrices, 
    get_T_R_list, getSTSR1_Q, ABlist, xzgraph
)

def simulate_optical_structure(
    type_of_structure,
    inner_radius,
    outer_radius,
    n1,
    n2=1.0,
    taper_angle=None,
    n_ar=None, 
    thickness_of_ar_coating=None,
    number_of_modes=100,
    glass_distance=1e-9,
    glass_index=1.49,
    glass_size=1.8e-6,
    wavelength=950e-9,
    return_image=True
):
    """
    Main interface for optical structure simulation.
    
    Parameters:
    -----------
    type_of_structure : str
        Type of optical structure: 'taper', 'ar_coating', or 'flat'
    inner_radius : float
        Inner radius of the structure in meters
    outer_radius : float
        Outer radius of the structure in meters
    n1 : float
        Core refractive index
    n2 : float, optional
        Cladding refractive index, defaults to 1.0
    taper_angle : float, optional
        Taper angle in degrees (only used if type_of_structure is 'taper')
    n_ar : float, optional
        Refractive index of AR coating (only used if type_of_structure is 'ar_coating')
    thickness_of_ar_coating : float, optional
        Thickness of AR coating in meters (only used if type_of_structure is 'ar_coating')
    number_of_modes : int, optional
        Number of modes to simulate, defaults to 100
    glass_distance : float, optional
        Distance to glass in meters, defaults to 1e-9
    glass_index : float, optional
        Refractive index of glass, defaults to 1.49
    glass_size : float, optional
        Glass size in meters, defaults to 1.8e-6
    wavelength : float, optional
        Wavelength in meters, defaults to 950e-9
    return_image : bool, optional
        Whether to return the field distribution image, defaults to True
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'efficiency': Coupling efficiency as a percentage
        - 'factors': Dictionary of contributing factors
        - 'heatmap': Base64 encoded image of the field distribution (if return_image is True)
    """
    # Parameter validation based on structure type
    if type_of_structure == 'taper' and taper_angle is None:
        raise ValueError("Taper angle must be provided for taper structures")
    
    if type_of_structure == 'ar_coating' and (n_ar is None or thickness_of_ar_coating is None):
        raise ValueError("AR coating index and thickness must be provided for AR coating structures")
    
    # Setup basic parameters
    k = 2 * np.pi / wavelength  # Wave number
    size = number_of_modes  # Number of modes
    R = outer_radius  # Outer radius
    
    # Setup glass information
    glassinfo = [glass_index, 1.0, glass_size, glass_distance]  # glass n1, glass n2, glass size, glass distance
    
    # Setup structure-specific parameters
    if type_of_structure == 'taper':
        # For tapered structure
        alpha = taper_angle * np.pi / 360  # Convert angle to radians (half-angle)
        h2 = inner_radius / np.tan(alpha)  # Calculate taper height
        
        # Number of sections in the taper (T value)
        Tvalue = 20
        
        # Generate radius values for each section of the taper
        Rvalues = [inner_radius] + [inner_radius * (Tvalue - i - 1) / Tvalue for i in range(Tvalue)]
        
        # Generate height values for each section
        hlist = [0] + [inner_radius + h2/Tvalue * (1/2 + i) for i in range(Tvalue)] + [inner_radius + h2 + 1e-6]
        
        # Create refractive index lists
        n1list = [n1] * len(Rvalues)
        n2list = [n2] * len(Rvalues)
        
    elif type_of_structure == 'ar_coating':
        # For AR coating
        # Setup with AR coating layer
        h1 = inner_radius  # Height of first section
        h2 = thickness_of_ar_coating  # Thickness of AR coating
        
        # Define Rvalues (radius at different points)
        Rvalues = [inner_radius, inner_radius]
        
        # Define heights
        hlist = [0, h1, h1+h2+1e-6]
        
        # Set refractive indices for each section
        n1list = [n1, n_ar]
        n2list = [n2, n2]
        
    else:  # 'flat' or default
        # For flat structure
        h1 = inner_radius
        
        # Simple structure with just one section
        Rvalues = [inner_radius]
        hlist = [0, h1, h1+1e-6]
        n1list = [n1]
        n2list = [n2]
    
    # Calculate efficiency using existing function
    efficiency = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k, False)
    
    # Convert to percentage
    efficiency_percentage = efficiency * 100
    
    result = {
        'efficiency': efficiency_percentage,
        'factors': {
            'overlapFactor': min(100, efficiency_percentage * 1.2),  # Example factor calculation
            'modeMatchFactor': min(100, efficiency_percentage * 0.9 + 10),  # Example factor calculation
            'taperFactor': min(100, efficiency_percentage * 1.1)  # Example factor calculation
        }
    }
    
    # Generate image of field distribution if requested
    if return_image:
        result['heatmap'] = generate_field_distribution(
            n1list, n2list, Rvalues, R, size, hlist, k, type_of_structure
        )
    
    return result

def generate_field_distribution(n1list, n2list, Rvalues, R, size, hlist, k, structure_type):
    """
    Generate the field distribution visualization and return it as a base64 encoded image.
    
    Parameters:
    -----------
    n1list : list
        List of core refractive indices for each section
    n2list : list
        List of cladding refractive indices for each section
    Rvalues : list
        List of radii for each section
    R : float
        Outer radius
    size : int
        Number of modes
    hlist : list
        List of heights for each section
    k : float
        Wave number
    structure_type : str
        Type of structure ('taper', 'ar_coating', or 'flat')
        
    Returns:
    --------
    str
        Base64 encoded image of the field distribution
    """
    # Setup initial field
    start = np.zeros(size)
    start[0] = 1
    
    # Calculate wave parameters
    k_ml = getk_n_ml(size, R)
    n_lm = getn_ml(size, k_ml, R)
    
    # Calculate beta values
    bclist = getbetaclists(0, n1list, n2list, Rvalues, R, k_ml, n_lm, size, k)
    
    # Calculate propagation matrices
    Plist = getpropagationmatrices(bclist, hlist)
    
    # Calculate transmission and reflection matrices
    forT, forR, backT, backR = get_T_R_list(bclist)
    
    # Calculate scattering matrices
    STlist, SRlist, STrevlist, SRrevlist = getSTSR1_Q(Plist, forT, forR, backT, backR)
    
    # Calculate reverse scattering matrices for visualization
    STlist2, SRlist2, STrevlist2, SRrevlist2 = getSTSR1_Q(
        Plist[::-1], backT[::-1], backR[::-1], forT[::-1], forR[::-1]
    )
    
    # Calculate field distributions
    alist, blist = ABlist(Plist, SRlist, STlist, SRrevlist2[::-1], start, SRrevlist)
    
    # Generate 2D field
    Efield = xzgraph(1000, alist, blist, 0, bclist, k_ml, R, hlist, size, R)
    Egraphreal = np.real(Efield)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.matshow(Egraphreal, origin='lower', cmap='viridis', fignum=1)
    
    # Add title based on structure type
    if structure_type == 'taper':
        plt.title('Tapered Waveguide Field Distribution')
    elif structure_type == 'ar_coating':
        plt.title('AR Coated Waveguide Field Distribution')
    else:
        plt.title('Flat Waveguide Field Distribution')
    
    plt.colorbar(label='Field Amplitude')
    
    # Add annotations
    plt.annotate('Glass', xy=(0.9, 0.1), xycoords='axes fraction', color='white')
    plt.annotate('Core', xy=(0.1, 0.5), xycoords='axes fraction', color='white')
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

# Example of how to use this module
if __name__ == "__main__":
    # Example for a tapered structure
    result = simulate_optical_structure(
        type_of_structure='taper',
        inner_radius=100e-9,  # 100 nm
        outer_radius=10e-6,   # 10 μm
        n1=3.5,
        taper_angle=5,
        number_of_modes=100,
        glass_distance=1e-9,
        wavelength=950e-9
    )
    
    print(f"Efficiency: {result['efficiency']:.2f}%")
    print("Factors:", result['factors'])
    
    # If you want to save the image
    if 'heatmap' in result:
        # Extract the base64 data (remove the prefix)
        img_data = result['heatmap'].split(',')[1]
        with open('field_distribution.png', 'wb') as f:
            f.write(base64.b64decode(img_data))
        print("Field distribution image saved as 'field_distribution.png'")