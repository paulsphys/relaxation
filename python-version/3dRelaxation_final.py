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
    val_arr = np.empty(r.shape)
    it = np.nditer(r, flags=['multi_index'])
    #Potential cutoff
    for i in it:
        if i < 0.33:
            val_arr[it.multi_index] = 10000
        else:
            A       = 0.5448504E6;
            epsilon = 10.8;
            alpha   = 13.353384 
            D       = 1.241314;
            C6      = 1.3732412;
            C8      = 0.4253785;
            C10     = 0.1781;

            x = i 

            Urep = A * np.exp(-alpha*x);

            ix2 = 1.0 / (x * x);
            ix6 = ix2 * ix2 * ix2;
            ix8 = ix6 * ix2;
            ix10 = ix8 * ix2;
            Uatt = -( C6*ix6 + C8*ix8 + C10*ix10 ) * F(x,D);
            val = epsilon * (Urep + Uatt)
            val_arr[it.multi_index] = val
    return val_arr
            


def PIMC_potential(r,R):
    val_arr = np.empty(r.shape)
    it = np.nditer(r, flags=['multi_index'])
    for i in it:
        if i > 2.35:
            val_arr[it.multi_index] = 1000
        else:
            epsilon = 1.4212942968311457
            sigma = 5.442152204855649/rm
            density = 0.0087*(rm**3)
            x = i / R;
            x2 = x*x;
            x4 = x2*x2;
            x6 = x2*x4;
            x8 = x4*x4;
            f1 = 1.0 / (1.0 - x2);
            sigoR3 = pow(sigma/R,3.0);
            sigoR9 = sigoR3*sigoR3*sigoR3;
            v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2) -
                    8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
            v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
            val = ((np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3))
            val_arr[it.multi_index] = val
    return val_arr
"""
def PIMC_potential(x,R):
    return -x**2 + (5.0/4)*(x**4)
"""
def W(x1,x2,y1,y2):
    #Kelvin
    R = 8.65/rm
    rho1 = np.sqrt(x1**2 + y1**2)
    rho2 = np.sqrt(x2**2 + y2**2)
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return (PIMC_potential(rho1,R) + PIMC_potential(rho2,R) + Aziz_potential(r))/E_rec
    #return (PIMC_potential(abs(x1),R) + PIMC_potential(abs(x2),R))/E_rec

#Cartesian Space
Lx = 18; Ly = 18; Lz = 25
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
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z,indexing='ij')

#Fourier Space
dkx = 2*np.pi/Lxnd; dky = 2*np.pi/Lynd; dkz = 2*np.pi/Lznd;
kxmin = -np.pi/dx; kxmax = np.pi/dx; kymin = -np.pi/dy; kymax = np.pi/dy; kzmin = -np.pi/dz; kzmax = np.pi/dz;
kx = dkx * np.concatenate(( np.arange(0,nx/2,1),np.arange(-nx/2,0,1) ))
ky = dky * np.concatenate(( np.arange(0,ny/2,1),np.arange(-ny/2,0,1) ))
kz = dkz * np.concatenate(( np.arange(0,nz/2,1),np.arange(-nz/2,0,1) ))
kx1, kx2, ky1, ky2, kzr = np.meshgrid(kx,kx,ky,ky,kz,indexing='ij')

#Laplacian in Fourier Space
fftminlap = kx1**2 + kx2**2 + ky1**2 + ky2**2 + 2*kzr**2

coff = 300/E_rec
"""
it = np.nditer(W(x1,x2,y1,y2), flags=['multi_index'])
Wcut = copy.deepcopy(W(x1,x2,y1,y2))
for i in it:
    if (i > coff or math.isnan(i)):
        Wcut[it.multi_index] = coff

"""


with open('3dRelax3dpot.npy','rb') as f:
    Wcut = np.load(f)        
"""
print(Wcut[1,1,1,1])
#Plot Potential part
#plt.contourf(x1,x2,Wcut,levels=100,vmax=10)
plt.pcolormesh(Wcut[36,:,36,:], vmax=20)
plt.colorbar()
plt.title('Full_potential')
plt.show()

"""
#Preconditioning-operator
fftPinv = 1./(coff + fftminlap)

#Define Pinv of a function
def Pinv(f):
    #return f
    return (np.fft.ifftn(np.multiply(fftPinv, np.fft.fftn(f))))
def P(f):
    return coff*f + np.fft.ifftn(np.multiply(fftminlap, np.fft.fftn(f)))

def symindex(phi):
    p1 = np.sum(phi**2, axis = (0,2))*dx*dy
    p2 = np.sum(phi**2, axis = (1,3))*dx*dy
    return np.sqrt(np.sum((p1-p2)**2))

#Iteration
dt = 1.2
center = 1.0
width = 0.6
s = 0.1
psi = np.exp((-(x1 + center)**2 - (x2 - center)**2 - (y1 + center)**2 -(y2 - center)**2)/(width**2))
psi = np.sqrt(2)*psi/np.sqrt(np.sum(psi**2)*dx*dx*dy*dy)
#print(np.sum(psi**2)*dx*dx*dy*dy)
A = np.sqrt(2/(4 + 2*np.sum(np.multiply(np.transpose(np.conj(psi),(2,3,0,1,4)),np.transpose(psi,(1,0,3,2,4))))*dx*dx*dy*dy*dz))
psi = A*(psi + np.transpose(psi,(1,0,3,2,4)))
print(np.sum(psi**2)*dx*dy*dx*dy)

erroreqs = 1
error = 1
i = 0
"""
#Plot 1st particle
p = np.sum(psi**2, axis = (1,3))*dx*dy
plt.contourf(x,y,p)
plt.colorbar()
plt.show()

#Plot 2nd particle
p = np.sum(psi**2, axis = (0,2))*dx*dy
plt.contourf(x,y,p)
plt.show()
"""
rec_error = np.array([])
eqs_error = np.array([])
eqs_error2 = np.array([])
energ = np.array([])
#symarray = np.array([symindex(psi)])
#symarrayafterupdate = np.array([])
xmsh, ymsh = np.meshgrid(x,y, indexing='ij')


while (error > 1e-6):
    i = i + 1
    print(i)
    psi_old = copy.deepcopy(psi)

    #Compute L0
    potentialpart = np.multiply(Wcut,psi)
    kinetic = np.fft.ifftn(np.multiply(fftminlap, np.fft.fftn(psi)))
    L00psin = kinetic + potentialpart
    num = np.sum(np.multiply(Pinv(psi),L00psin))
    den = np.sum(np.multiply(Pinv(psi),psi))
    E1 = num/den
    L0psin = L00psin - E1*psi

    if (i > 1):
        L0psislow = L0psin - L0psinold

    if (i > 100):

        L0psislow = L0psin - L0psinold
        alpha = np.sum(np.multiply(psislow,L0psislow))/np.sum(np.multiply(psislow,P(psislow)))
        #alpha_err = np.append(alpha_err, alpha)
        gamma = min(1 - s/(alpha*dt),0)
        #subtracted_mode = 0
        subtracted_mode = gamma*psislow*((np.sum(np.multiply(psislow,L0psin)))/(np.sum(np.multiply(psislow,P(psislow)))))
    else:
        subtracted_mode = np.zeros(psi.shape)

    #E2 = np.sum(np.multiply(psi,L00psin))/(np.sum(np.multiply(psi,psi)))
    #erroreqs = np.sqrt(np.sum((L00psin -  E2*psi)**2))
    
    psi = psi - dt*(Pinv(L0psin) - subtracted_mode)
    psi = np.sqrt(2)*psi/np.sqrt(np.sum(psi**2)*dx*dx*dy*dy*dz)
    #symarrayafterupdate = np.append(symarrayafterupdate, symindex(psi_n))
    #symarray = np.append(symarray, symindex(psi_n))
    A = np.sqrt(2/(4 + 2*np.sum(np.multiply(np.transpose(np.conj(psi),(2,3,0,1,4)),np.transpose(psi,(1,0,3,2,4))))*dx*dx*dy*dy*dz))
    psi = A*(psi + np.transpose(psi,(1,0,3,2,4)))
    print(np.sum(psi**2)*dx*dx*dy*dy*dz)
    #psi_n = (0.5)*(psi_n + np.transpose(np.conj(psi_n),axes = (1,0,3,2)))
    psislow = psi - psi_old
    L0psinold = L0psin

    """
    plt.plot(x,psi[:,40,40,40])
    plt.show()
    plt.contourf(x,y,kinetic[:,40,:,40])
    plt.colorbar()
    plt.show()
    plt.contourf(x,y,potentialpart[:,40,:,40])
    plt.colorbar()
    plt.show()
    """
    #symarray = np.append(symarray, symindex(psi_n))
    error = np.sqrt(np.sum(psi - psi_old)**2)
    rec_error = np.append(rec_error, error)
    plt.semilogy(rec_error,'r+')
    E2 = np.sum(np.multiply(psi,L00psin))/(np.sum(np.multiply(psi,psi)))
    erroreqs = np.sqrt(np.sum((L00psin - E2*psi)**2))
    #erroreqs2 = np.sqrt(np.sum((Pinv(L00psin - E2*psi)**2))) 
    eqs_error = np.append(eqs_error, erroreqs)
    #eqs_error2 = np.append(eqs_error2, erroreqs2)
    energ = np.append(energ, E1)
    plt.semilogy(eqs_error,'b*')
    #plt.semilogy(eqs_error2,'g*')
    if (i % 100 == 0):
        
        #plt.show()
        plt.savefig( "Error_3d_long" + str(i) + ".pdf" )
        plt.close()
        """          
        #Plot Potential1stparticle
        p = np.sum(potentialpart, axis = (1,3))*dx*dy
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #ax.plot_surface(xmsh,ymsh,p)
        #plt.contourf(x,y,p,cmap = cm.Blues, alpha=0.5)
        #plt.colorbar()
        plt.plot(p[78,:])
        plt.show()
        plt.contourf(x,y,psi[:,40,:,40])
        plt.colorbar()
        plt.show()

        #Plot Potential2ndparticle
        p = np.sum(potentialpart, axis = (0,2))*dx*dy
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #ax.plot_surface(xmsh,ymsh,p)
        #plt.contourf(x,y,p,cmap=cm.Reds, alpha = 0.5)
        #plt.colorbar()
        plt.plot(p[:,2])
        plt.show()
        """
        #with open('Energy_sym3.npy','wb') as f:
        #    np.save(f,energ)
        #plt.semilogy(energ,'r+')
        #plt.savefig("Energy_unsym3_"+str(i) + ".pdf")
        #plt.show()
        #plt.close()
        #print(symarray)
        #plt.semilogy(symarrayafterupdate,'b+')
        #plt.savefig("symindexafterupdate2"+str(i)+".pdf")
        #plt.close()
        #plt.semilogy(symarray,'b+')
        #plt.savefig("symindexunsymacc"+str(i)+".pdf")
        #plt.show()
        #plt.close()
        """
        #Plot 1st particle
        p = np.real(np.sum(psi_n**2, axis = (1,3))*dx*dy)
        plt.contourf(x,y,p,cmap=cm.Blues,alpha=0.5)
        plt.colorbar()
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #ax.plot_surface(xmsh,ymsh,p)
        plt.savefig( "Particle1_unsym_" + str(i) + ".pdf")
        plt.close()
        #Plot 2nd particle
        p = np.sum(psi_n**2, axis = (0,2))*dx*dy
        plt.contourf(x,y,p,cmap=cm.Reds,alpha=0.5)
        plt.colorbar()
        #fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        #ax.plot_surface(xmsh,ymsh,p)
        plt.savefig( "Particle2_unsym3_" + str(i) + ".pdf")
        plt.close()
        #Plot Difference
        diff = np.sum(psi_n - psi,axis=(0,2))*dx*dy       
        plt.contourf(diff)
        plt.colorbar()
        #fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
        #ax.plot_surface(xmsh,ymsh,p)
        plt.savefig( "Diff_sym3_" + str(i) + ".pdf")
        plt.close()
        """
    

print('Converged after ' + str(i) + ' iterations.')
print(E1)
print(E2)
with open('Final_wavefunction_3d_c300_bigger.npy','wb') as f:
    np.save(f,psi)


