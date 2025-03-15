from commands import *
import os
import numpy as np
import pandas as pd
output_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import openpyxl
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
    
        
def variantAR(h1,alphadegrees,rWG,ARh,k,startdist,enddist,n):
    glass_distances = np.linspace(startdist, enddist, num=n)
    effectivities = []
    # glassinfo[3] = 2e-6
    # effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,True)
    
    for ARhvar in glass_distances:
        glassinfo = [1.4949,1.4533,1.8e-6,1e-9] #n1,n2,glasssize,glass distance
        alpha = alphadegrees*np.pi/360
        h2 = rWG/np.tan(alpha)
        Rvalues = [rWG,rWG,rWG]# len 21
        hlist = [0,h1,h1+ARhvar,h1+2*ARhvar] # len 22
        n1list = [1.44,1.2,1]
        n2list = [1]*len(Rvalues)
        effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,False)
        print(f'rWG: {ARhvar}, Effectivity: {effectivity}')
        effectivities.append(effectivity)
    print(glass_distances,effectivities)
    df = pd.DataFrame({'Glass Distance': glass_distances, 'Effectivity': effectivities})
    output_path = os.path.join(output_dir, 'Variant_Distances.xlsx')
    with pd.ExcelWriter(output_path, mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=f'AR_R{rWG}', index=False)

wavelength = 950e-9
k = 2*np.pi/wavelength
graphsize = 3000
R = 6e-6
graphR = 6e-6
rWG = 100e-9
size = 100

# for rWGhere in [1e-7,3e-7,5e-7,7e-7,9e-7]:
    # variantAR(1e-6,5,rWGhere,5,k,1e-7,4e-6,100)
# Load the workbook and get the sheet names
# workbook_path = os.path.join(output_dir, 'Variant_Distances.xlsx')
# workbook = openpyxl.load_workbook(workbook_path)
# sheet_names = workbook.sheetnames

# # Initialize lists to store data
# x_axis = []
# data_points = []
# sheet_names = [f'AR_R{rWGhere}' for rWGhere in [1e-7,3e-7,5e-7,7e-7,9e-7]]

# # Iterate through each sheet and extract data
# for sheet_name in sheet_names:
#     sheet = workbook[sheet_name]
#     if not x_axis:
#         x_axis = [cell.value for cell in sheet['A'][1:]]  # First column as x-axis
#     data_points.append([cell.value for cell in sheet['B'][1:]])  # Second column as data points

# # Plot the data
# print(len(x_axis))
# for i, data in enumerate(data_points):
#     plt.plot(x_axis, data, label=sheet_names[i])

# plt.xlabel('Glass Distance')
# plt.ylabel('Effectivity')
# plt.legend()
# plt.show()
size = 50
variantAR(rWG,5,1e-6,20,k,1e-18,2e-6,2)

# while i < 5:
#     i+= 1

# f = [1,2,34,2,]
# for i in range(5) == [0,12,3,4]:
    