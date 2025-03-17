from OpticSimProj.Workspace.commands import *
import os
import numpy as np
import pandas as pd
output_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import openpyxl

def Pcalc(betac):
    beta = betac[0][0]
    c = betac[1][0]
    sumc = 0
    for mode in c:
        sumc += abs(mode)**2
    return np.real(beta*sumc)
def getEffectivity(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,getgraph):
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
        Egraphreal = np.real(xzgraph(1000, alist,blist, 0, bclist, k_ml, R,hlist,size,R))
        plt.matshow(Egraphreal, origin='lower')
        plt.colorbar()
        plt.show()
    t2 = abs(a_q[0])**2
    return t2*Pcalc(bclist[-1])/Pcalc(bclist[0])

def fixsizeR(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,effectivity,inf):
    bettereff = 0
    topeff = effectivity
    notR = True
    while notR:
        bettereffR = getEffectivity(n1list, n2list, Rvalues, R*1.1, size, hlist, glassinfo, k,False)
        effdiffR = bettereffR/effectivity
        if effdiffR > 1.05 or effdiffR < 0.95:
            R *= 1.1
            inf['R'] = R
            print(f'Error: Effectivity is not stable, increasing R to: {R}')
        else:
            notR = False
    notsize = True
    while notsize:
        bettereff = getEffectivity(n1list, n2list, Rvalues, R, size+10, hlist, glassinfo, k,False)
        effdiff = bettereff/effectivity
        if effdiff > 1.05 or effdiff < 0.95:
            size += 10
            inf['size'] = size
            print(f'Error: Effectivity is not stable, increasing size to: {size}')
        else:
            notR = False
        if size >150:
            print('Error: Size is too large, effectivity is not stable')
            fail = True
            return R,size,0
            break
    return R,size,bettereff

def isfail(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,effectivity,inf,tolerance):
    bettereffR = getEffectivity(n1list, n2list, Rvalues, R*tolerance[0], size+tolerance[1], hlist, glassinfo, k,False)
    effdiffR = bettereffR/effectivity
    print(f'Efficiency difference:{((effdiffR-1)*100):.4f}%')
    return abs(effdiffR-1) > tolerance[2]

def onevar(info,changevar,startdist,enddist,n,tolerance):
    testvals = np.linspace(startdist, enddist, num=n)
    effectivities = []
    fail = False
    testindex = 0
    inf = info.copy() #copies info because mutable
    
    for varval in testvals:
        if fail == True:
            effectivities.append(0)
        else:
            inf[changevar] = varval
            n1,n2,R,rWG,size,alphadegrees,h1,h3,Tvalue,nglass,nfiber,glasssize,glassdistance,k = [inf[i] for i in inf]

            Tvalue = int(Tvalue)
            glassinfo = [nglass,nfiber,glasssize,glassdistance] #n1,n2,glasssize,glass distance
            alpha = alphadegrees*np.pi/360 #changes to radians
            h2 = rWG/np.tan(alpha)
            if Tvalue == 1:
                Rvalues = [rWG,0,glasssize] # len 2
                hlist = [0,h1,h1+glassdistance,2*h1+glassdistance] # len 5
            else:
                Rvalues = [rWG]+[rWG*(Tvalue-i-1)/Tvalue for i in range(Tvalue)]+[glasssize] # len 21
                hlist = [0]+[h1 + h2/Tvalue*(i)for i in range(Tvalue)] + [h1+h2+glassdistance]+[h1+h2+h3+glassdistance] # len 22
            n1list  = ((len(Rvalues)-2)*[n1])+[1]+[nglass]
            n2list  = ((len(Rvalues)-1)*[n2])+[nfiber]
            
            effectivity = getEffectivity(n1list, n2list, Rvalues, R, size, hlist, glassinfo, k,showgraph)
            print(changevar + f': {varval:.3e}, Effectivity: {100*effectivity:.2f}%')
            if testindex == 0 and tolerance[2]:
                if isfail(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,effectivity,inf,tolerance):
                    fail = True
                    effectivity = 0
                    print('Error: Effectivity is not stable, setting rest to 0')
                # R,size, effectivity= fixsizeR(n1list,n2list,Rvalues,R,size,hlist,glassinfo,k,effectivity,inf)
                # if effectivity == 0:
                #     fail = True
                #     print('Error: Effectivity is not stable, setting rest to 0')
            effectivities.append(effectivity)
        testindex += 1
        if testindex == int(n**(1/2)):
            testindex = 0
            fail = False
    return effectivities

def twovarchanges(info,changevar,startdist,enddist,n,changevar2,startdist2,enddist2,n2,backtest):
    testvals = np.linspace(startdist, enddist, num=n)
    effectivitylists = []
    nameslist = []
    inf = info.copy() #copies info because mutable
    for varval in testvals:
        inf[changevar] = varval
        effectivities = onevar(inf,changevar2,startdist2,enddist2,n2,backtest)
        print('Finished with ' + changevar + f': {varval:.3e}')
        effectivitylists.append(effectivities)
        nameslist.append(f'{changevar}: {varval}')
    return effectivitylists,nameslist

def testandsave(info,changevar,startdist,enddist,n,changevar2,startdist2,enddist2,n2,tolerance,filename):
    efflist,namelist = twovarchanges(info,changevar,startdist,enddist,n,changevar2,startdist2,enddist2,n2,tolerance)
    df = pd.DataFrame(efflist, index=namelist)
    df.columns = [f'{changevar2}: {val}' for val in np.linspace(startdist2, enddist2, num=n2)]
    output_path = os.path.join(output_dir, 'Finaldata',filename +'.csv')
    df.to_csv(output_path, index=True)

def get_x_axis(data):
    return [float(x.split(': ')[1]) for x in data]
def plot_data(file_name,plotname,xname):
    file_path = os.path.join(output_dir, 'Finaldata',file_name +'.csv')
    df = pd.read_csv(file_path, index_col=0)
    plt.figure(figsize=(10, 8))
    x_axis_name = 0
    x_axis_vals = 0
    for index, row in df.iterrows():
        if not x_axis_name:
            x_axis_name = df.index[0].split(': ')[0]
            x_axis_vals = get_x_axis(row.index)
        plt.scatter(x_axis_vals, row.values, label=index)
    plt.xlabel(xname)
    plt.ylabel('Effectivity')
    plt.title(plotname)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()

wavelength = 950e-9

info = { #initializing vals
    'n1': 3.5,
    'n2': 1,
    'Outer Radius': 10e-6,
    'Inner Radius': 1000e-9,
    'Number of modes': 150,
    'Taper Angle': 90,
    'Height to Taper': 2e-6,
    'h3': 2e-6,
    'Tvalue': 1,
    'Glass n value': 1.4949,
    'Fiber n value': 1.4533,
    'glasssize': 1.8e-6,
    'Distance to Glass': 1e-9,
    'k': 2*np.pi/wavelength
}
tolerance = [1.2,100,0.1]#Rmult,sizeadd,tolerance (set tolerance to 0 if not needed)
showgraph = False 
filename = 'test' #Dglass_InnerRadius is over effs
# testandsave(info,'Inner Radius',1e-8,1e-5,5,'Taper Angle',1,10,100,tolerance,filename)
testandsave(info,'Distance to Glass',1e-8,2e-6,3,'Inner Radius',1e-6,2e-6,4,tolerance,filename)
plot_data(filename,'Effectivity Relative to Inner Radius','Inner Radius') #filename, plotname, x-axis name
