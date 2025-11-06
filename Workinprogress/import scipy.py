import scipy
import numpy as np
import pylab as p
from numpy import linspace
from numpy import array
from scipy import pi,sin,cos,tan,sqrt,arctan,arccos
from PIL import Image
import matplotlib as mpl
from operator import itemgetter
import pylab
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

import os
from matplotlib import rc

filename = 'V1710_Sco.txt'  #change filename here

x = np.loadtxt(filename, delimiter=',', dtype = '<f8', usecols=[0])
z = np.loadtxt(filename, delimiter=',',dtype = '<f8', usecols=[1])
err = 0.01*z #np.loadtxt(filename, delimiter=',', dtype = '<f8', usecols=[2])
y = np.loadtxt(filename, delimiter=',', dtype = 'S15', usecols=[4])
print(y)

t0 = 2459318.13211  # change the date here to t0 in JD

x = x - t0

nV = 322  # change the number here to the number of available V observations
#to do this, after you download the file, use ctl + f to search for a specific letter/word. Then search for ,v, (make sure to put commas before and after).
#this should give you the number of V band data in the file. change nV accordingly. Same for CV and Vis.

nCV = 25  # change the number here to the number of available CV observations
nVis = 57 # change the number here to the number of available Vis observations

nUF = 0
nUL = 3
nB = 298
nAN = 862

V = [0 for V in range(nV)]
Vx = [0 for Vx in range(nV)]
Verr = [0 for Verr in range(nV)]

CV = [0 for CV in range(nCV)]
CVx = [0 for CVx in range(nCV)]
CVerr = [0 for CVerr in range(nCV)]

Vis = [0 for Vis in range(nVis)]
Visx = [0 for Visx in range(nVis)]
Viserr = [0 for Viserr in range(nVis)]

UF = [0 for UF in range(nUF)]
UFx = [0 for UFx in range(nUF)]
UFerr = [0 for UFerr in range(nUF)]

UL = [0 for UL in range(nUL)]
ULx = [0 for ULx in range(nUL)]
ULerr = [0 for ULerr in range(nUL)]

B = [0 for B in range(nB)]
Bx = [0 for Bx in range(nB)]
Berr = [0 for Berr in range(nB)]

AN = [0 for AN in range(nAN)]
ANx = [0 for ANx in range(nAN)]
ANerr = [0 for ANerr in range(nAN)]

v=0
j=0
h=0
k=0
m=0
f=0
c=0

a = len(y)


for i in range(a):

    if y[i] == b'V':
        V[v] = z[i]
        Vx[v] = x[i]
        Verr[v] = err[i]
        v = v + 1           
    elif y[i] == b'CV':
        CV[j] = z[i]
        CVx[j] = x[i]
        CVerr[j] = err[i]
        j = j + 1

    elif y[i] == b'Vis.':
        Vis[h] = z[i]
        Visx[h] = x[i]
        Viserr[h] = 0# err[i]
        h = h + 1
                                        
    elif y[i] == b'UF':
        UF[k] = z[i]
        UFx[k] = x[i]
        UFerr[k] = err[i]
        k = k + 1
        
    elif y[i] == b'UL':
        UL[m] = z[i]
        ULx[m] = x[i]
        ULerr[m] =  err[i]
        m = m + 1 
    elif y[i] == b'B':
        B[f] = z[i]
        Bx[f] = x[i]
        Berr[f] =  err[i]
        f = f + 1  
    elif y[i] == b'AN':
        AN[c] = z[i]
        ANx[c] = x[i]
        ANerr[c] =  err[i]
        c = c + 1
                
        
ax = plt.subplots (figsize=(11,6.4))       
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().invert_yaxis()

ax.errorbar(Visx, Vis, yerr = 0, fmt='rd',  markersize=5, label = "Vis.")
ax.errorbar(CVx, CV, yerr = 0, fmt='bo',  markersize=5, label = "$\it{CV}$")
#ax.errorbar(UFx, UF, yerr = 0, fmt='ko',  markersize=8, label = "$\it{\;Unfiltered}$")
ax.errorbar(ULx, UL, yerr = 0, fmt='kv',  markersize=7)
#ax.errorbar(Bx, B, yerr = 0, fmt='cs',  markersize=9,label = "B")
ax.errorbar(ANx, AN, yerr = 0, fmt='mo',  markersize=6, label = "ASAS-SN")
#ax.errorbar(4.8, 10.526, xerr = 0.6948838713,fmt='g',markersize=8,capsize=4,linewidth=2)
ax.errorbar(Vx, V, yerr = 0, fmt='g*',  markersize=8, label = "$\it{\;V}$")

Vmax = [8.526,8.526] #insert Vmax here - change both values into Vmax
maxdate = 2459318.13211 # insert the date of max in JD
maxdate = maxdate - t0
tmax =[maxdate,maxdate]
t2 = [4.4944,4.4944] #insert t2 in days here - change both values into t2
tmaxt2 = [0,0]
tmaxt2[0] = tmax[0] + t2[0]
tmaxt2[1] = tmax[1] + t2[1]
Vmaxt2=[0,0]
Vmaxt2[0] = Vmax[0] + 2.0
Vmaxt2[1] = Vmax[1] + 2.0
ymax = [0,25]
xmax = [-10,300]
ax.plot(tmax, ymax, 'k--', linewidth = 1.0)
ax.plot(tmaxt2, ymax, 'k--', linewidth = 1.0)
ax.plot(xmax, Vmax, 'k--', linewidth = 1.0)
ax.plot(xmax, Vmaxt2, 'k--', linewidth = 1.0)
############################################

matplotlib.rcParams.update({'font.size': 20})  
                                     
minor_xticks = np.arange(-50, 600, 5)                                               
minor_yticks = np.arange(0, 19.0, 0.5)  
  
ax.set_yticks(minor_yticks, minor=True)                                                       
ax.set_xticks(minor_xticks, minor=True) 

ax.set_xlim(-8,63.5)  #adjust the upper limit accordingly
ax.set_ylim(18,8) #adjust both limits accordingly
            
plt.tick_params(width=3,length=8)
plt.tick_params(which='minor', width=2,length=4)
ax.set_xlabel('Days since peak magnitude ', fontsize = 20)
ax.set_ylabel('Magnitude', fontsize = 20)

ax.legend(bbox_to_anchor=(0.2,0.1), loc=1, borderaxespad=0.)
ax.legend(numpoints=1)
plt.text(50,9.2,'V1710 Sco')

ax.plot(2459308.622-t0,17.570,marker='v', color = 'm', markersize=7)
ax.plot(2459309.437-t0,17.506,marker='v', color = 'm', markersize=7)
ax.plot(2459309.679-t0,17.505,marker='v', color = 'm', markersize=7)
ax.plot(2459310.721-t0,17.737,marker='v', color = 'm', markersize=7)
ax.plot(2459310.794-t0,17.558,marker='v', color = 'm', markersize=7)
ax.plot(2459310.835-t0,17.701,marker='v', color = 'm', markersize=7)
ax.plot(2459311.849-t0,17.369,marker='v', color = 'm', markersize=7)
ax.plot(2459311.86-t0,17.524,marker='v', color = 'm', markersize=7)
ax.plot(2459313.467-t0,17.674,marker='v', color = 'm', markersize=7)
ax.plot(2459313.698-t0,17.616,marker='v', color = 'm', markersize=7)
ax.plot(2459314.714-t0,17.700,marker='v', color = 'm', markersize=7)
ax.plot(2459315.674-t0,17.608,marker='v', color = 'm', markersize=7)
ax.plot(2459315.704-t0,17.729,marker='v', color = 'm', markersize=7)
ax.plot(2459315.896-t0,17.570,marker='v', color = 'm', markersize=7)
ax.plot(2459314.22014-t0,11,marker='v', color = 'r', markersize=7)
ax.plot(2459320.7618-t0,9.9,marker='v', color = 'r', markersize=7)
ax.plot(2459324.22569-t0,11,marker='v', color = 'r', markersize=7)

tpp = 2459318.66224-t0
tpn = t0-2459317.924086
t2p = 2459322.922459-(t2[0]+t0)
t2n = (t2[0]+t0)-2459321.923709
errorp = np.sqrt((t2p)**2+(tpp)**2)
errorn = np.sqrt((t2n)**2+(tpn)**2)


print("The pos error is",errorp)
print("The neg error is",errorn)
print("The tpeak pos error is",tpp)
print("The tpeak neg error is",tpn)
print("The t2 pos error is",t2p)
print("The t2 neg error is",t2n)



t2_lower_bound = t2n*(-1) +(t2[0])
t2_upper_bound = t2p +(t2[0])
ylimits = plt.gca().get_ylim()
plt.fill_between([ t2_lower_bound , t2_upper_bound ] , ylimits[0] , ylimits[1] , color = "black" , alpha = 0.25 , zorder = 0)
plt.ylim(ylimits)

tpeak_lower_bound = tpn*(-1)+(maxdate)
tpeak_upper_bound = tpp+(maxdate)
ylimits = plt.gca().get_ylim()
plt.fill_between([ tpeak_lower_bound , tpeak_upper_bound ] , ylimits[0] , ylimits[1] , color = "black" , alpha = 0.25 , zorder = 0)
plt.ylim(ylimits)

#plt.plot(Vx, V, '-ob')
#import scipy.interpolate
#x_interp = scipy.interpolate.interp1d(V, Vx)
#print(x_interp(10.526))

plt.tight_layout()
plt.savefig('V1710_Sco_LC.pdf')  #Change name here
plt.close()