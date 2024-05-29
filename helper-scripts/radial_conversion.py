"""
Code for getting the radial distribution from a cartesian probability distribution. Used for comparing results from the relaxation code and pimc.
Adapted from here - https://stackoverflow.com/questions/41708681/regrid-3d-cartesian-to-polar-netcdf-data-with-python 
"""

import numpy as np
import copy
import scipy.integrate as intgr
import scipy.interpolate as intrp
import matplotlib.pyplot as plt 


rm = 2.9673;

def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simps(x*psi, x)
   print (int_psi_square)
   return int_psi_square

def radial_ave(ψ,X,Y):
    
    x = X[0,:]
    y = Y[:,0]
    N = X.shape[0]
    dx = X[1,0]-X[0,0]
    dR = np.sqrt(2*dx**2)
    dϕ = 2*π/N
    r_vals = np.arange(0, R, dR)
    ϕ_vals = np.arange(0, 2*π, dϕ)

    if len(r_vals)*len(ϕ_vals) > N**2:
        print("Warning: Oversampling")

    # Initialize data on polar grid with fill values
    fill_value = -9999.0
    data_polar = fill_value*np.ones((len(r_vals), len(ϕ_vals)))

    # Define radius of influence. A nearest neighbour outside this radius will not
    # be taken into account.
    radius_of_influence = np.sqrt(0.1**2 + 0.1**2)

    # For each cell in the polar grid, find the nearest neighbour in the cartesian
    # grid. If it lies within the radius of influence, transfer the corresponding
    # data.
    for r, row_polar in zip(r_vals, range(len(r_vals))):
        for ϕ, col_polar in zip(ϕ_vals, range(len(ϕ_vals))):
            # Transform polar to cartesian
            _x = r*np.cos(ϕ)
            _y = r*np.sin(ϕ)

            # Find nearest neighbour in cartesian grid
            d = np.sqrt((_x-X)**2 + (_y-Y)**2)
            nn_row_cart, nn_col_cart = np.unravel_index(np.argmin(d), d.shape)
            dmin = d[nn_row_cart, nn_col_cart]

            # Transfer data
            if dmin <= radius_of_influence:
                data_polar[row_polar, col_polar] = ψ[nn_row_cart, nn_col_cart]

    # Mask remaining fill values
    data_polar = np.ma.masked_equal(data_polar, fill_value)
    
    return r_vals, np.average(data_polar,axis=1)
   
#Compute Recoil Energy
hbar = 11606  * 6.5821 * 10**(-16) #Kelvin second
hbar2by2 = hbar ** 2/2 #Kelvin^2 second^2
mass = 4* 9.3827 * 11606 * 10**8 #Kelvin/c^2
c = 2.998 * 10**(18) #Angstrom per second
E_rec = hbar2by2/(mass*rm**2)*c**2
print (E_rec)

Lx = 18; Ly = 18; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72;
ny = 72;
nz = 8;
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')
#psi = np.exp(-(x1 + 1.0)**2/2.0 -(y1 + 1.0)**2/2.0 - (x2 - 1.0)**2/2.0 - (y2 - 1.0)**2/2.0)
nrm = dx*dy*np.sqrt(dz*rm)*rm*rm
with open('Final_wavefunction_3d_c300_big.npy', 'rb') as f:
	psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print("Bare norm")
print(psif.shape)
print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
#p = np.mean(p, axis = 2)
#plt.pcolor(p)
#plt.show()

print(np.sum(p)*(rm**3)*dx*dy*dz)
#plt.pcolormesh(p[:,:,1])
#plt.colorbar()
#plt.show()
rho_rad = []
π = np.pi
R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
norm = normalize_psi_PIMC(radial,rval)
plt.plot(rval,radial,'g*')

f2 = open('radial-N-reduce-ba6781ba-73f7-47a9-926b-ac1ba1ec2880.dat','r')
lines = f2.readlines()
x = np.array([])
y = np.array([])
z = np.array([])
for line in lines[3:]:
    p = line.split()
    x = np.append(x,float(p[0]))
    y = np.append(y,float(p[1]))
    z = np.append(z,float(p[2]))
f2.close()


norm = normalize_psi_PIMC(y,x)
plt.errorbar(x,y, yerr=z,linestyle="None",marker = 'x', color='red',label='PIGS, -t 0.005')
"""
Older code used for conversion (introduces oscillations in the radial distribution):

f2 = open('ce-position-00.033-0002-00.000-0.00500-ba6781ba-73f7-47a9-926b-ac1ba1ec2880.dat','r')
lines = f2.readlines()
p = np.array([])
sep = lines[1].split()
dx = float(sep[4])
dy = float(sep[7])
dz = float(sep[10])
for line in lines[3:]:
    spl = line.split()
    p = np.append(p,float(spl[0]))

f2.close()


num_grid_sep = 51
p = np.reshape(p,(51,51,51))
p = np.mean(p,axis=2)
plt.pcolor(p)
plt.show()

R = 8.65
dR = np.sqrt(dx**2 + dy**2)
N_R = int(np.floor(R/dR))+1
norm = np.ones(N_R)
rvals = np.arange(0,R-0*dR,dR)

for n in range(N_R):
    norm[n] /= np.pi*(2*n+1)*dR*dR
    
ρ_rad = np.zeros([num_grid_sep,N_R])
for iz in range(num_grid_sep):
    for i in range(num_grid_sep):
        y = -R + dy*i + dy/2
        for j in range(num_grid_sep):
            x = -R + dx*j + + dx/2
            r = np.sqrt(x**2 + y**2)
            k = int(r/dR)           
            if k==0:
                pass
                #print(f'z = {-L/2 + iz*δ[-1] + δ[-1]/2,x,y,i,j}')
            if k < N_R:                
                ρ_rad[iz,k] += p[i,j,iz]*dx*dy

ρ_rad *= norm
print (2*np.pi*25*intgr.simps(rvals*np.mean(ρ_rad,axis=0), rvals))
plt.plot(rvals, np.mean(ρ_rad,axis=0), 'b*')
"""
plt.show()
