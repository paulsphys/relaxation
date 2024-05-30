import numpy as np
import math
import copy
import scipy.special as scp
import scipy.optimize as opt
import scipy.integrate as intgr
import matplotlib.pyplot as plt
from matplotlib import cm

rm = 2.9673;

#Compute Recoil Energy
hbar = 11606  * 6.5821 * 10**(-16) #Kelvin second
hbar2by2 = hbar ** 2/2 #Kelvin^2 second^2
mass = 4* 9.3827 * 11606 * 10**8 #Kelvin/c^2
c = 2.998 * 10**(18) #Angstrom per second
E_rec = hbar2by2/(mass*rm**2)*c**2
print (E_rec)

def F(x,D):
    a = np.zeros(np.shape(x))
    it = np.nditer(x, flags=['multi_index'])
    for i in it:
        if (i < D):
            a[it.multi_index]  = np.exp(-(D/i - 1.0)*(D/i - 1.0))
        else:
            a[it.multi_index] = 1.0
    return a

def Aziz_potential(r):
    A       = 0.5448504E6;
    epsilon = 10.8;
    alpha   = 13.353384 
    D       = 1.241314;
    C6      = 1.3732412;
    C8      = 0.4253785;
    C10     = 0.1781;

    x = r

    Urep = A * np.exp(-alpha*x);

    ix2 = 1.0 / (x * x);
    ix6 = ix2 * ix2 * ix2;
    ix8 = ix6 * ix2;
    ix10 = ix8 * ix2;
    Uatt = -( C6*ix6 + C8*ix8 + C10*ix10 ) * F(x,D);
    return ( epsilon * (Urep + Uatt) );

def PIMC_potential(r,R):
    epsilon = 1.4212942968311457
    sigma = 5.442152204855649/rm
    density = 0.0087*(rm**3)
    x = r / R;
    x2 = x*x;
    x4 = x2*x2;
    x6 = x2*x4;
    x8 = x4*x4;
    f1 = 1.0 / (1.0 - x2);
    sigoR3 = pow(sigma/R,3.0);
    sigoR9 = sigoR3*sigoR3*sigoR3;
    v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2) - 8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
    v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
    val = (np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3)
    return val


def W(x1,x2,y1,y2,zr):
    #Kelvin
    coff = 300/E_rec
    R = 8.65/rm
    rho1 = np.sqrt(x1**2 + y1**2)
    rho2 = np.sqrt(x2**2 + y2**2)
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + zr**2)
    piece1 = PIMC_potential(rho1,R)/E_rec
    piece1[np.isnan(piece1)] = coff + 200
    piece1[piece1 > coff] = coff 
    piece2 = PIMC_potential(rho2,R)/E_rec
    piece2[np.isnan(piece2)] = coff + 200
    piece2[piece2 > coff] = coff
    piece3 = Aziz_potential(r)/E_rec
    piece3[np.isnan(piece3)] = coff + 200
    piece3[piece3 > coff] = coff
    """
    plt.pcolormesh(piece1[:,40,:,40,1,2])
    plt.colorbar()
    plt.show()

    plt.pcolormesh(piece2[:,40,:,40,1,2])
    plt.colorbar()
    plt.show()
    
    plt.pcolormesh(piece3[:,40,:,40,1,2])
    plt.colorbar()
    plt.show()
    
    plt.pcolormesh(piece3[:,40,:,40,1,1])
    plt.colorbar()
    plt.show()
    
    plt.plot(x,piece3[:,40,40,40,1,1])
    plt.show()

    plt.plot(x,piece3[:,40,40,40,1,2])
    plt.show()
    """
    #with open('3dRelax2dPIMCpot.npy','wb') as f:
    #    np.save(f, piece1)
    
    #with open('3dRelax2dPIMCpot.npy','wb') as f:
    #    np.save(f, piece2)

    #with open('3dRelax2dAziz.npy','wb') as f:
    #    np.save(f, piece3)
    
    return piece1 + piece2 + piece3

#Cartesian Space
Lx = 18; Ly = 18; Lz = 25;
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72;
ny = 72;
nz = 12;
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2,nz/2,1)

coff = 300/E_rec
x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z,indexing='ij')
Wpot =  W(x1,x2,y1,y2,zr)
plt.pcolormesh(x,y,Wpot[:,36,:,36,1])
plt.colorbar()
plt.show()
Wpot[Wpot > coff] = coff
"""
xt,yt = np.meshgrid(x,y,indexing='ij')
plt.pcolormesh(xt,yt,PIMC_potential(np.sqrt(xt**2 + yt**2), 8.65/rm))
plt.colorbar()
plt.show()
"""
with open('3dRelax3dpot.npy','wb') as f:
    np.save(f,Wpot)

print('Saved to file')

