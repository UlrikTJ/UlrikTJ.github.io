import base64
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.optimize import root_scalar
from scipy.linalg import eig, inv
import io

def send_pic(image_path="cat.png"):
    # Read the image file in binary mode and encode it in base64
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    image_path = os.path.join(script_dir, image_path)  # Construct the full path to the image
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    # Return the base64 encoded string
    return encoded_string

c = 3e8
mu0 = 4*np.pi*10**(-7)
def N_lmcalc(m:int,k_ml,R:float):
    return -R**2/2*jv(m-1, k_ml[m]*R)*jv(m+1, k_ml[m]*R)

def N_lmmatrix(size,k_ml,R):
    N_lm = np.zeros((size,size))
    for i in range(len(N_lm)):
        N_lm[i] = N_lmcalc(i,k_ml,R)
    return N_lm

def find_bessel_zeros(m:int, max_zeros, max_range=10000):
    zeros = []
    initial_guesses = np.arange(1, max_range, 1)  # Guesses in range 1 to max_range for the zeros
    for guess in initial_guesses:
        def bessel_for_rootfinding(x):
            return jv(m, x)
        if bessel_for_rootfinding(guess) * bessel_for_rootfinding(guess + 1) < 0:
            sol = root_scalar(bessel_for_rootfinding, bracket=[guess, guess + 1], method='brentq')
            if sol.converged and len(zeros) < max_zeros:
                zeros.append(sol.root)
            elif len(zeros) >= max_zeros:  # Stop once we find enough zeros
                break
    return zeros

def k_mlFind(size,R):
    ret = np.zeros((size,size))
    for m in range(len(ret)):
        ret[m] = np.array(find_bessel_zeros(m,size))/R
    return ret

def jinte0(r:float,R:float,l1,l2,m:int,k_ml):
    if l1 == l2:
        return (R**2/2*(jv(m,k_ml[m][l1]*R)**2-(jv(m-1,k_ml[m][l1]*R)*jv(m+1,k_ml[m][l1]*R))))-(r**2/2*(jv(m,k_ml[m][l1]*r)**2-(jv(m-1,k_ml[m][l1]*r)*jv(m+1,k_ml[m][l1]*r))))
    else:
        return R*(k_ml[m][l2]*jv(m-1,k_ml[m][l2]*R)*jv(m,k_ml[m][l1]*R)-k_ml[m][l1]*jv(m-1,k_ml[m][l1]*R)*jv(m,k_ml[m][l2]*R))/(k_ml[m][l1]**2-k_ml[m][l2]**2)-r*(k_ml[m][l2]*jv(m-1,k_ml[m][l2]*r)*jv(m,k_ml[m][l1]*r)-k_ml[m][l1]*jv(m-1,k_ml[m][l1]*r)*jv(m,k_ml[m][l2]*r))/(k_ml[m][l1]**2-k_ml[m][l2]**2)

def O2(l1,l2,m:int,n1,n2,r:float,R:float,k_ml,N_lm,k):
    return k**2/(np.sqrt(N_lm[m][l1]*N_lm[m][l2]))*(n1**2*jinte0(0,r,l1,l2,0,k_ml)+n2**2*jinte0(r,R,l1,l2,0,k_ml))

def Ncalc(l,m:int,R,k_ml):
    return -R**2/2*jv(m-1,k_ml[m][l]*R)*jv(m+1,k_ml[m][l]*R)
    
def fBasis(l,m:int,r:float,R:float,k_ml):  
    return jv(m, k_ml[m][l]*r)/np.sqrt(Ncalc(l,m,R,k_ml))

def big0finder(m:int,size,n1,n2,r:float,R:float,k_ml,N_lm,k):
    BigO = np.zeros((size,size))
    Dia = [-(k_ml[m][l])**2 for l in range(size)]
    for i in range(len(N_lm[m])):
        for j in range(len(N_lm[m])):
            BigO[i,j] = O2(i,j,m,n1,n2,r,R,k_ml,N_lm,k)
            if i==j:
                BigO[i,j] += Dia[i]
    return BigO


def beta2cfromfinal0(finalO):
    eigenvalues, eigenvectors = eig(finalO)
    eigsorted = np.sort(eigenvalues)
    eigfuncsorted = eigenvectors[:,np.argsort(eigenvalues)]
    eigvaldec = eigsorted[::-1] #beta^2 list
    eigfuncdec = eigfuncsorted[:,::-1] #c list
    return eigvaldec, eigfuncdec.T

def getabetaclist(mlist,size,n1,n2,rWG,R,k_ml,N_lm,k):
    # betalist = [0]*(max(mlist)+1)
    # clist = [0]*len(betalist)
    # for i in mlist:
    betalist, clist = beta2cfromfinal0(big0finder(mlist,size,n1,n2,rWG,R,k_ml,N_lm,k))
    return betalist,clist

def ecalc(m,lmax,r,j,R,clist,k_ml):
    tot = 0
    for l in range(lmax):
        tot += jv(m,k_ml[m][l]*r)/np.sqrt(Ncalc(l,m,R,k_ml))*clist[m][j][l]
    return tot

def sqrtbeta(betalis):
    a = betalis.copy()
    for i in range(len(betalis)):
        a[i] = np.sqrt(betalis[i])
    return a
def PropogationMatrix(betalist,deltaz):
    size = len(betalist)
    Pmatrix = np.zeros((size,size),dtype=complex)
    for i in range(size):
        Pmatrix[i][i] = np.exp(1j*betalist[i]*deltaz)
    return Pmatrix
def T_R_calc(bclist, i,dir): #start index and direction, +1,-1
    c1 = bclist[i][1].T
    c2 = bclist[i+dir][1].T
    B1matrix = np.diag(bclist[i][0])
    B2matrix = np.diag(bclist[i+dir][0])
    ulrik =  c2 @ B2matrix @ np.linalg.inv(c2)  
    Rval = np.linalg.inv(c1 @ B1matrix + ulrik @ c1) @ (c1 @ B1matrix - ulrik @ c1)
    Tval = np.linalg.inv(c2) @ c1 + np.linalg.inv(c2) @ c1 @ Rval
    return Rval,Tval

def getbetaclists(m,n1list,n2list,rWGlist,R,k_ml,Nlm,size,k): #for a specific m
    bclistfinal = [0]*len(n1list)
    for i in range(len(n1list)):
        betalist,clist = getabetaclist([m],size,n1list[i],n2list[i],rWGlist[i],R,k_ml,Nlm,k)
        bclistfinal[i] = [sqrtbeta(betalist[m]),clist[m]]
    return bclistfinal
def f2calc(m, l, r, k_ml,R):
    return jv(m, k_ml[m][l] * r) / Ncalc(l, m, R, k_ml)
def optimizedzlayer(dim,clist,blist,m,a,b,z,botz,topz,f_vals):
    expthing = np.matrix([np.exp(1j * blist[j] * (z-botz))*a[j] + np.exp(1j * blist[j] * (topz-z))*b[j] for j, coeff in enumerate(a)])
    Ejmz = np.zeros(dim, dtype=complex)
    mult = np.cos(m*np.pi)
    ecalclist = clist@f_vals.T
    Ejmz[dim//2:] = ((ecalclist.T).dot(expthing.T)).flatten()
    Ejmz[dim//2-1::-1] = mult*Ejmz[dim//2::]
    return Ejmz
def xzgraph(dim, alist,blist, m, bclist, k_ml, R,hlist,size,graphR):
    z_vals = np.linspace(hlist[0], hlist[-1], dim)
    x_vals = np.linspace(-graphR, graphR, dim)
    f_vals = np.array([[f2calc(m, l, r, k_ml,R) for l in range(size)] for r in x_vals[dim//2::]])
    Ejm = np.zeros((dim, dim), dtype=complex)
    ind = 0
    for i, z in enumerate(z_vals):
        if z > hlist[ind+1]:
            ind += 1
        Ejm[i] = optimizedzlayer(dim,bclist[ind][1],bclist[ind][0],m,alist[ind],blist[ind],z,hlist[ind],hlist[ind+1],f_vals)
    return Ejm
def getpropagationmatrices(bclist, hlist):
    size = len(bclist)
    matrices = [0]*size
    for i in range(size):
        matrices[i] = PropogationMatrix(bclist[i][0], (hlist[i+1]-hlist[i]))
    return matrices
def get_T_R_list(bclist):
    size = len(bclist)-1
    forT,forR,backT,backR = [0]*size,[0]*size,[0]*size,[0]*size
    for i in range(size):
        forR[i],forT[i] = T_R_calc(bclist,i,1)
        backR[i],backT[i] = T_R_calc(bclist,i+1,-1)
    return forT,forR,backT,backR
def SRT1_3(R21,R12,T21,P2,R23,T12,T23): #R21,R12,T21,P2,R23,T12,T23
    ICMatrix= inv(np.identity(len(R12[0])) - R21 @ P2 @ R23 @ P2)
    return R12 + T21@P2@R23@P2@ICMatrix@T12, T23@P2@ICMatrix@T12
def getSTSR1_Q(Plist,forT,forR,backT,backR):
    STlist,SRlist,STrevlist,SRrevlist = [0]*len(forT),[0]*len(forT),[0]*len(forT),[0]*len(forT) #gives Ri1 kinda stuff for rev
    STlist[0], SRlist[0],STrevlist[0],SRrevlist[0] = forT[0],forR[0],backT[0],backR[0]
    for i in range(len(forT)-1):
        P2,R12,R23,T12,T23,T21,R21= Plist[i+1],SRlist[i],forR[i+1],STlist[i],forT[i+1],STrevlist[i],SRrevlist[i]
        SRlist[i+1],STlist[i+1] = SRT1_3(R21,R12,T21,P2,R23,T12,T23) #R12,T21,P2,R23,T12
        SRrevlist[i+1],STrevlist[i+1] = SRT1_3(R23,backR[i+1],T23,P2,R21,backT[i+1],T21)
    return STlist,SRlist,STrevlist,SRrevlist
def ABQ(Pq,SRq1,SRqQ,ST1q,a1):
    a = inv(np.identity(len(Pq))-SRq1@Pq@SRqQ@Pq)@ST1q@a1
    b = inv(np.identity(len(Pq))-SRqQ@Pq@SRq1@Pq)@SRqQ@Pq@ST1q@a1
    return a,b
def ABlist(Plist,SRlist,STlist,SRrevlist,start,SRrevlist3):
    size = len(Plist)
    a1 = Plist[0]@start
    a,b = [0]*size,[0]*size
    a[0],b[0],a[-1],b[-1] = start, SRlist[-1]@a1, STlist[-1]@a1, np.zeros(len(a1))
    for i in range(1,size-1):
        a[i],b[i] = ABQ(Plist[i],SRrevlist3[i-1],SRrevlist[i],STlist[i-1],a1)
    return a,b
def get_filename(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'Graphs/Tapered')
    png_dir = os.path.join(output_dir, 'Epsfiles/'+file_name+'.eps')
    eps_dir = os.path.join(output_dir, 'Pngfiles/'+file_name+'.png')
    return png_dir, eps_dir

def save_files(png_dir, eps_dir,Egraphreal):
    plt.imsave(png_dir, Egraphreal, origin='lower', cmap='viridis')
    plt.imsave(eps_dir, Egraphreal, origin='lower', cmap='viridis')
    
def getEgraph(n1list,n2list,Rvalues,R,size,hlist,k,graphsize,graphR):
    start = np.zeros(size)
    start[0] = 1
    k_ml = getk_n_ml(size,R)
    n_lm = getn_ml(size,k_ml,R) #save this
    bclist  = getbetaclists(0,n1list,n2list,Rvalues,R,k_ml,n_lm,size,k) #21 of em #save bclists
    Plist   = getpropagationmatrices(bclist,hlist) #21 of em
    forT,forR,backT,backR = get_T_R_list(bclist) #20 of em #1s 
    STlist,SRlist,STrevlist,SRrevlist  = getSTSR1_Q(Plist,forT,forR,backT,backR) #20 of em
    STlist2,SRlist2,STrevlist2,SRrevlist2  = getSTSR1_Q(Plist[::-1],backT[::-1],backR[::-1],forT[::-1],forR[::-1]) #20 of em
    alist,blist = ABlist(Plist,SRlist,STlist,SRrevlist2[::-1],start,SRrevlist)
    Egraph= xzgraph(graphsize, alist,blist, 0, bclist, k_ml, R,hlist,size,graphR)
    return Egraph

def getk_n_ml(size,R):
    # output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saveddata', 'bclists', f'k_n_ml_{size}_{R}.txt')
    # if not os.path.exists(output_file):
    #     f = k_mlFind(size,R)
    #     np.savetxt(output_file, f)
    # else:
    #     f = np.loadtxt(output_file)
    return k_mlFind(size,R)
def getn_ml(size,k_ml,R):
    # output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saveddata', 'bclists', f'n_ml_{size}_{R}.txt')
    # if not os.path.exists(output_file):
    f = N_lmmatrix(size,k_ml,R)      
    #     np.savetxt(output_file, f)
    # else:
    #     f = np.loadtxt(output_file)
    return f
def getbetaclists(m,n1list,n2list,rWGlist,R,k_ml,Nlm,size,k): #for a specific m
    bclistfinal = [0]*len(n1list)
    for i in range(len(n1list)):
        betalist,clist = getbc_lists(m,size,n1list[i],n2list[i],rWGlist[i],R,k_ml,Nlm,k)
        bclistfinal[i] = [sqrtbeta(betalist),clist]
    return bclistfinal
def getbc_lists(m,size,n1,n2,rWG,R,k_ml,Nlm,k)  :
    # output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saveddata', 'bclists',f'b_lists_{m}_{size}_{rWG}_{R}_{n1}_{n2}_{k}.txt')
    # output_file2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saveddata', 'bclists',f'c_lists_{m}_{size}_{rWG}_{R}_{n1}_{n2}_{k}.txt')
    # if not os.path.exists(output_file):
    #     b,c = getabetaclist(m,size,n1,n2,rWG,R,k_ml,Nlm,k)    
    #     np.savetxt(output_file, b)
    #     np.savetxt(output_file2, c)
    # else:
    #     b = np.loadtxt(output_file, dtype=complex)
    #     c = np.loadtxt(output_file2, dtype=complex)\
    b,c = getabetaclist(m,size,n1,n2,rWG,R,k_ml,Nlm,k)    
    return b,c

def Pcalc(betac):
    beta = betac[0][0]
    c = betac[1][0]
    sumc = 0
    for mode in c:
        sumc += abs(mode)**2
    return np.real(beta*sumc)

def getEffectivity(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,getgraph):
    n1list += [glassinfo[0]]
    n2list += [glassinfo[1]]
    Rvalues += [glassinfo[2]]
    hlist[-1] = hlist[-2]+glassinfo[3]#last hvalue wont matter anymore but should still exist
    hlist += [hlist[-1]*1.1]
    start = np.zeros(size)
    start[0] = 1
    k_ml = getk_n_ml(size,R)
    n_lm = getn_ml(size,k_ml,R) #save this
    bclist  = getbetaclists(0,n1list,n2list,Rvalues,R,k_ml,n_lm,size,k) #21 of em #save bclists
    Plist   = getpropagationmatrices(bclist,hlist) #21 of em
    forT,forR,backT,backR = get_T_R_list(bclist) #20 of em #1s 
    STlist,SRlist,STrevlist,SRrevlist  = getSTSR1_Q(Plist,forT,forR,backT,backR) #20 of em
    a_q = STlist[-1]@Plist[0]@start
    t2 = abs(a_q[0])**2
    Effectivity = t2*Pcalc(bclist[-1])/Pcalc(bclist[0])
    if getgraph:
        STlist2,SRlist2,STrevlist2,SRrevlist2  = getSTSR1_Q(Plist[::-1],backT[::-1],backR[::-1],forT[::-1],forR[::-1]) #20 of em
        alist,blist = ABlist(Plist,SRlist,STlist,SRrevlist2[::-1],start,SRrevlist)
        Egraphreal = np.real(xzgraph(1000, alist,blist, 0, bclist, k_ml, R,hlist,size,R))
        plt.matshow(Egraphreal, origin='lower', extent=[-R * 1e6, R * 1e6, hlist[0] * 1e6, hlist[-1] * 1e6], aspect='auto')
        plt.xlabel('Width (μm)')
        plt.ylabel('Height (μm)')
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        # plt.matshow(Egraphreal, origin='lower')
        # plt.colorbar()
        # plt.show()
        return Effectivity, Egraphreal
    return Effectivity

def simulate(info):
    Effectivity, Egraphreal = getEffectivity(info, True)
    # Encode Egraphreal into base64

    # Convert the Egraphreal matrix to an image
    fig, ax = plt.subplots()
    cax = ax.matshow(Egraphreal, origin='lower', cmap='viridis')
    plt.colorbar(cax)

    # Save the image to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image in base64
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Return or print the encoded image
    return Effectivity, encoded_image

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
    efficiency, heatmap = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k, return_image)
    
    # Convert to percentage
    efficiency_percentage = efficiency * 100
    
    result = {
        'efficiency': efficiency_percentage,
        'factors': {
            'overlapFactor': min(100, efficiency_percentage * 1.2),  # Example factor calculation
            'modeMatchFactor': min(100, efficiency_percentage * 0.9 + 10),  # Example factor calculation
            'taperFactor': min(100, efficiency_percentage * 1.1)  # Example factor calculation
        },
        'heatmap': heatmap
    }
    return result

info = { #initializing vals
    'n1': 3.5,
    'n2': 1,
    'Outer Radius': 10e-6,
    'Inner Radius': 1000e-9,
    'Number of modes': 50,
    'Taper Angle': 90,
    'Height to Taper': 2e-6,
    'h3': 2e-6,
    'Tvalue': 1,
    'Distance to Glass': 1e-9,
    'wavelength': 1.55e-6,
    'fiber': 'UNHA3'
}
if __name__ == "__main__":
    getEffectivity(info, True)