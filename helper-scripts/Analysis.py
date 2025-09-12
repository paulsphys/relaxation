import numpy as np
import copy
import scipy.integrate as intgr
import scipy.interpolate as intrp
import scipy.special as scp
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser(description='Takes the radial density and produces various properties')

parser.add_argument('--filename','-f',required = True)
parser.add_argument('-R',type = float, help = "The radius of the pore", required = True)
parser.add_argument('--Element', '-E', help = "Element", required = True)

parsed_args = vars(parser.parse_args())
rm = 2.9673;

def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simps(x*psi, x)
   print ("Norm = " + str(int_psi_square))
   return int_psi_square

filename =  parsed_args["filename"]
#filename = "Radial-wavefunction-" + fend + ".npz"
with np.load(filename) as f:
    rval = f['arr_0']
    radial = f['arr_1']
R = parsed_args["R"]
norm = normalize_psi_PIMC(radial, rval)
meanr = 2*np.pi*25*intgr.simps(rval*rval*radial, rval)/norm
spatial_ext = np.sqrt(2*np.pi*25*intgr.simps(rval*((rval - meanr)**2)*radial, rval)/norm)
print('Spatial extent = ' + str(spatial_ext))
print('Mean r = ' + str(meanr))
print('Mean rbyR = ' + str(meanr/R))
OneHe = 1.4
indx = tuple([rval <=  OneHe])
indx2 = tuple([rval <= 2*OneHe])
indx3 = tuple([rval > 2*OneHe])

OneHeExt = 2*np.pi*25*intgr.simpson(rval[indx]*radial[indx], rval[indx])/norm
TwoHeExt = 2*np.pi*25*intgr.simpson(rval[indx2]*radial[indx2], rval[indx2])/norm
RestHeExt = 2*np.pi*25*intgr.simpson(rval[indx3]*radial[indx3], rval[indx3])/norm

print('One He Radius = ' + str(OneHeExt))
print('Two He Radius = ' + str(TwoHeExt))
print('Rest He Radius = ' + str(RestHeExt))

def PIMC_potential(r,R,eps,sig,den):
    epsilon = eps
    sigma =  sig/rm
    density =  den*(rm**3)
    x = r / R;
    x2 = x*x;
    x4 = x2*x2;11.76 - 8
    x6 = x2*x4;
    x8 = x4*x4;
    f1 = 1.0 / (1.0 - x2);
    sigoR3 = pow(sigma/R,3.0);19.75
    sigoR9 = sigoR3*sigoR3*sigoR3;
    v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2) - 8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
    v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
    val = (np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3)
    return val

Element = {
        'Cs': [5.44,1.359,0.0091], 
        'Ar':  [3.0225,36.136,0.0265], 
        'Au':  [3.305,19.59,0.0595], 
        'K': [5.14,1.512,0.0139], 
        'Mg': [3.885,5.661,0.0437], 
        'Rb': [5.417,1.251,0.0114], 
        'Ne': [2.695,19.75,0.0440]
}

E = parsed_args["Element"]
sigma = Element[E][0]
epsilon = Element[E][1]
density = Element[E][2]
r = np.linspace(0,R-0.1,500)

pot = PIMC_potential(r,R, epsilon, sigma, density)
idx = np.argmin(pot)
Welldepth = r[idx]

rlimit = Welldepth + 2*OneHe
llimit = Welldepth - 2*OneHe
indx = []
for i in rval:
    if (i >= llimit and i <= rlimit):
        indx.append(True)
    else:
        indx.append(False)
indx = tuple([indx])
Wellint = 2*np.pi*25*intgr.simpson(rval[indx]*radial[indx], rval[indx])/norm

print('Integral in the well is = ' + str(Wellint))
