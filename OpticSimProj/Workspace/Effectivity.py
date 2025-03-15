from commands import *
import os
import numpy as np
import pandas as pd
output_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
    if getgraph:
        STlist2,SRlist2,STrevlist2,SRrevlist2  = getSTSR1_Q(Plist[::-1],backT[::-1],backR[::-1],forT[::-1],forR[::-1]) #20 of em
        alist,blist = ABlist(Plist,SRlist,STlist,SRrevlist2[::-1],start,SRrevlist)
        Egraphreal = np.real(xzgraph(1000, alist,blist, 0, bclist, k_ml, R,hlist,size,graphR))
        plt.matshow(Egraphreal, origin='lower')
        plt.colorbar()
        plt.show()
    return abs(a_q[0])**2

wavelength = 950e-9
k = 2*np.pi/wavelength
graphsize = 3000
R = 6e-6
graphR = 6e-6
size = 100
rWG = 100e-9
def variantglassSize(h1,alphadegrees,rWG,Tvalue,k,startdist,enddist,n):
    glassinfo = [1.4949,1.4533,1.8e-6,1e-6] #n1,n2,glasssize,glass distance
    alpha = alphadegrees*np.pi/360
    h2 = rWG/np.tan(alpha)
    Rvalues = [rWG]+[rWG*(Tvalue-i-1)/Tvalue for i in range(Tvalue)]# len 21
    hlist = [0]+[h1 + h2/Tvalue*(1/2+i)for i in range(Tvalue)] + [h1+h2+1e-6] # len 22
    n1list = [1.44]*len(Rvalues)
    n2list = [1]*len(Rvalues)

    glass_distances = np.logspace(np.log10(startdist), np.log10(enddist), num=n)
    effectivities = []
    glassinfo[3] = 2e-6
    # effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,True)
    
    for glass_distance in glass_distances:
        glassinfo[2] = glass_distance
        effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,False)
        print(f'Glass Distance: {glass_distance}, Effectivity: {effectivity}')
        effectivities.append(effectivity)
    print(glass_distances,effectivities)
    
def variantDistance(h1,alphadegrees,rWG,Tvalue,k,startdist,enddist,n):
    glass_distances = np.linspace(startdist, enddist, num=n)
    effectivities = []
    # glassinfo[3] = 2e-6
    # effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,True)
    
    for glass_distance in glass_distances:
        glassinfo = [1.4949,1.4533,1.8e-6,1e-6] #n1,n2,glasssize,glass distance
        alpha = alphadegrees*np.pi/360
        h2 = rWG/np.tan(alpha)
        Rvalues = [rWG]+[rWG*(Tvalue-i-1)/Tvalue for i in range(Tvalue)]# len 21
        hlist = [0]+[h1 + h2/Tvalue*(1/2+i)for i in range(Tvalue)] + [h1+h2+1e-6] # len 22
        n1list = [1.44]*len(Rvalues)
        n2list = [1]*len(Rvalues)
        glassinfo[3] = glass_distance
        effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,False)
        print(f'Glass Distance: {glass_distance}, Effectivity: {effectivity}')
        effectivities.append(effectivity)
    print(glass_distances,effectivities)
    df = pd.DataFrame({'Glass Distance': glass_distances, 'Effectivity': effectivities})
    output_path = os.path.join(output_dir, 'Variant_Distances.xlsx')
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=f'variant_gd_size_finish_{size}', index=False)
    # plt.plot(glass_distances, effectivities)
    # plt.xscale('log')
    # plt.yscale('linear')
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    # plt.xlabel('Glass Distance (m)')
    # plt.ylabel('Effectivity')
    # plt.title('Effectivity vs Glass Distance')
    # plt.show()

def variantR(h1,alphadegrees,rWG,Tvalue,k,startR,endR,n):
    glassinfo = [1.4949,1.4533,1.8e-6,1e-6] #n1,n2,glasssize,glass distance
    alpha = alphadegrees*np.pi/360
    h2 = rWG/np.tan(alpha)
    Rvalues = [rWG]+[rWG*(Tvalue-i-1)/Tvalue for i in range(Tvalue)]# len 21
    hlist = [0]+[h1 + h2/Tvalue*(1/2+i)for i in range(Tvalue)] + [h1+h2+1e-6] # len 22
    n1list = [1.44]*len(Rvalues)
    n2list = [1]*len(Rvalues)

    rvals = np.linspace(np.log10(startR), np.log10(endR), num=n)
    effectivities = []
    glassinfo[3] = 2e-6
    R = rvals[0]
    # effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,True)
    
    for rval in rvals:
        R = rval
        effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,False)
        print(f'R: {R}, Effectivity: {effectivity}')
        effectivities.append(effectivity)
    print(rvals,effectivities)
    df = pd.DataFrame({'Glass Distance': rvals, 'Effectivity': effectivities})
    output_path = os.path.join(output_dir, 'Newsvaeddata.xlsx')
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=f'variantR_size_{size}', index=False)
        
def variantSmallr(h1,alphadegrees,rWG,Tvalue,k,startdist,enddist,n):
    glass_distances = np.linspace(startdist, enddist, num=n)
    effectivities = []
    # glassinfo[3] = 2e-6
    # effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,True)
    
    for rWG in glass_distances:
        glassinfo = [1.4949,1.4533,1.8e-6,1e-6] #n1,n2,glasssize,glass distance
        alpha = alphadegrees*np.pi/360
        h2 = rWG/np.tan(alpha)
        Rvalues = [rWG]+[rWG*(Tvalue-i-1)/Tvalue for i in range(Tvalue)]# len 21
        hlist = [0]+[h1 + h2/Tvalue*(1/2+i)for i in range(Tvalue)] + [h1+h2+1e-6] # len 22
        n1list = [1.44]*len(Rvalues)
        n2list = [1]*len(Rvalues)
        effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,False)
        print(f'rWG: {rWG}, Effectivity: {effectivity}')
        effectivities.append(effectivity)
    print(glass_distances,effectivities)
    df = pd.DataFrame({'Glass Distance': glass_distances, 'Effectivity': effectivities})
    output_path = os.path.join(output_dir, 'Variant_Distances.xlsx')
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=f'variant_gd_size_finish_{size}', index=False)
    # plt.plot(rvals, effectivities)
    # plt.xscale('log')
    # plt.yscale('linear')
    # plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    # plt.xlabel('Glass Distance (m)')
    # plt.ylabel('Effectivity')
    # plt.title('Effectivity vs Glass Distance')
    # plt.show()

    # a =getEgraph(n1list,n2list,Rvalues,R,size,hlist,k,graphsize,graphR)
    # Egraphreal = np.real(a)
    # plt.matshow(Egraphreal, origin='lower')
    # plt.colorbar()  # Add a colorbar to the plot  
    # plt.show()
# sizes = [25,100,200]
# for i in sizes:
#     size = i
#     variantDistance(rWG,5,1e-6,20,k,1e-8,2e-6,100)
size = 100
variantSmallr(rWG,5,1e-6,20,k,1e-7,4e-6,200)
# size = 50
# variantSmallr(rWG,5,1e-6,20,k,1e-18,2e-6,200)

# while i < 5:
#     i+= 1

# f = [1,2,34,2,]
# for i in range(5) == [0,12,3,4]:
    