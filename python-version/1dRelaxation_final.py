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
E_rec = hbar2by2/(mass*(rm**2))*(c**2)
print (E_rec)

#Define Potential function W
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
    val = epsilon * (Urep + Uatt)
    return val
            


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
    v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2)-8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
    v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
    val = ((np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3))
    return val

def W(x1,x2):
    #Kelvin
    R = 8.65/rm
    return (PIMC_potential(abs(x1),R) + PIMC_potential(abs(x2),R) + Aziz_potential(abs(x1-x2)))/E_rec
    

#Cartesian Space
Lx = 25;
Lxnd = Lx/rm;

nx = 1000;
dx = Lxnd/nx

xmin = -Lxnd/2; xmax = Lxnd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))

x1, x2 = np.meshgrid(x,x,indexing='ij')

#Fourier Space
dkx = 2*np.pi/Lxnd; 
kxmin = -np.pi/dx; kxmax = np.pi/dx;
kx = dkx * np.concatenate(( np.arange(0,nx/2,1),np.arange(-nx/2,0,1) ))

kx1, kx2 = np.meshgrid(kx,kx,indexing='ij')

#Laplacian in Fourier Space
fftminlap = kx1**2 + kx2**2

coff = 300/E_rec

Wcut = W(x1,x2)
Wcut[np.isnan(Wcut)] = coff + 200
Wcut[Wcut > coff] = coff

#Preconditioning-operator
fftPinv = 1./(coff + fftminlap)

#Define Pinv of a function
def Pinv(f):
    #return f
    piece1 = np.fft.fftn(f)
    #print(piece1)
    piece2 = np.multiply(fftPinv, piece1)
    #print(piece2)
    piece3 = np.real(np.fft.ifftn(np.multiply(fftPinv, np.fft.fftn(f))))
    #print(piece3)
    return piece3

def P(f):
    return coff*f + np.real(np.fft.ifftn(np.multiply(fftminlap, np.fft.fftn(f))))

#Iteration
dt = 1.0
s = 0.5
center = 1.0
width = 0.6
psi = np.exp((-(x1 + center)**2 - (x2 - center)**2)/(width**2))

psi = np.sqrt(2)*psi/np.sqrt(np.sum(psi**2)*dx*dx)

print(np.sum(psi**2)*dx*dx)
#psi = (1/np.sqrt(2))*(psi + psi.T)

A = np.sqrt(2/(4 +2* np.sum(np.multiply(np.conj(psi),psi.T))*dx*dx))
psi = A*(psi + psi.T)
print(np.sum(psi**2)*dx*dx)
#psi = np.sqrt(2)*psi/np.sqrt(np.sum(psi**2)*dx*dx)

error = 1
erroreqs = 1
i = 0
L0psinold = 0
"""
#Plot 1st particle
p = np.sum(psi**2, axis = 1)*dx
#plt.plot(x,p)
#plt.show()

#Plot 2nd particle
p = np.sum(psi**2, axis = 0)*dx
#plt.plot(x,p)
#plt.show()
"""

rec_error = np.array([])
eqs_error = np.array([])
alpha_err = np.array([])
while (erroreqs > 1e-5):
    i = i + 1

    if (i != 1): 
        #L0psislow = L0psin - L0psinold
        L0psinold = L0psin 
        psi = copy.deepcopy(psi_n)
    
    
    potentialpart = np.multiply(Wcut,psi)
    L00psin = np.real(np.fft.ifftn(np.multiply(fftminlap, np.fft.fftn(psi)))) + potentialpart
    num = np.sum(np.multiply(Pinv(psi),L00psin))
    den = np.sum(np.multiply(Pinv(psi),psi))
    
    E1 = num/den
    L0psin = L00psin - E1*psi
    
    if (i > 100):
        
        L0psislow = L0psin - L0psinold
        alpha = np.sum(np.multiply(psislow,L0psislow))/np.sum(np.multiply(psislow,P(psislow)))
        #print(alpha)
        alpha_err = np.append(alpha_err, alpha)
        gamma = min(1 - s/(alpha*dt),0)
        #subtracted_mode = 0 
        subtracted_mode = gamma*psislow*((np.sum(np.multiply(psislow,L0psin)))/(np.sum(np.multiply(psislow,P(psislow)))))
        
    else:
        subtracted_mode = np.zeros(psi.shape)

    psi_n = psi - dt*Pinv(L0psin) + subtracted_mode*dt
    #psi_n = (1/np.sqrt(2))*(psi_n + np.conj(psi_n).T)
    psi_n = np.sqrt(2)*psi_n/np.sqrt(np.sum(psi_n**2)*dx*dx)
    #print(np.sum(psi_n**2)*dx*dx)
    #Calculate normalisation
    #A = np.sqrt(2/(4 + 2*np.sum(np.multiply(np.conj(psi_n),psi_n.T))*dx*dx))
    #psi_n = A*(psi_n + psi_n.T)
    #print(np.sum(psi**2)*dx*dx)
    psislow = psi_n - psi

    error = np.sqrt(np.sum(psi_n - psi)**2)
    rec_error = np.append(rec_error, error)
    plt.semilogy(rec_error,'r+')
    E2 = np.sum(np.multiply(psi,L00psin))/(np.sum(np.multiply(psi,psi)))
    erroreqs = np.sqrt(np.sum((L00psin -  E2*psi)**2))
    eqs_error = np.append(eqs_error, erroreqs)
    plt.semilogy(eqs_error,'b*')

    if (False):
        
        plt.show()

               
        """
        plt.plot(psi_n[0,:])
        plt.plot(psi_n[-1,:])
        plt.plot(psi_n[:,0])
        plt.plot(psi_n[:,-1])
        plt.show()
        
        line1 = np.array([])
        line2 = np.array([])
        line3 = np.array([])
        line4 = np.array([])
        line5 = np.array([])
        
        for k in range(50,200):
            line1 = np.append(line1, psi_n[k,k])  
            line2 = np.append(line2, psi_n[k+1,k-1])
            line3 = np.append(line3, psi_n[k-1,k+1])
            line4 = np.append(line4, psi_n[k+2,k-2])
            line5 = np.append(line5, psi_n[k-2,k+2])
        plt.plot(line1, label="Diagonal")
        plt.plot(line2, label="Odb1")
        plt.plot(line3, label="Odb1os")
        plt.plot(line4, label="Odb2")
        plt.plot(line5, label="Odb2os")
        plt.legend()
        plt.show()
        
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x1, x2, psi_n)
        plt.show()
         
        plt.contourf(x,x,psi_n,levels=100)
        plt.colorbar()
        plt.show()

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x1, x2, potentialpart)
        plt.show()

        
        #Plot Potential1stparticle
        p = np.sum(potentialpart, axis = 1)*dx
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #ax.plot_surface(xms,ymsh,p)
        plt.plot(x,p)
        plt.show()

        #Plot Potential2ndparticle
        p = np.sum(potentialpart, axis = 0)*dx
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        plt.plot(x,p)
        plt.show()
        

        #Plot 1st particle
        p = np.sum(psi_n**2, axis = 1)*dx
        plt.plot(x,p)

        #Plot 2nd particle
        p = np.sum(psi_n**2, axis = 0)*dx
        plt.plot(x,p)
        plt.show()
        
        #Plot Difference
        diff = np.real(psi_n - psi)       
        plt.pcolormesh(diff)
        plt.colorbar()
        plt.show()
        
        plt.pcolormesh(np.real(subtracted_mode))
        plt.colorbar()
        plt.show()
        """ 

print(E2)
print(i)
print(error)
#plt.show()

"""
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.pcolormesh(np.real(psi_n))
plt.show()

#Plot 1st particle
p = np.sum(psi_n**2, axis = 1)*dx
plt.plot(x,p)

#Plot 2nd particle
p = np.sum(psi_n**2, axis = 0)*dx
plt.plot(x,p)
plt.show()
"""
