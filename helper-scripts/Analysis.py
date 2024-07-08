import numpy as np
import copy
import scipy.integrate as intgr
import scipy.interpolate as intrp
import matplotlib.pyplot as plt 
import argparse

parser = argparse.ArgumentParser(description='Creates the radial distribution from the full wavefunction data')

parser.add_argument('--filename','-f',required = True)
parser.add_argument('-R',type = float, help = "The number of grid_points in the x-y plane", required = True)
parser.add_argument('-n',type = int, help = "The number of grid_points in the x-y plane", default = 72)
parser.add_argument('--nz',type = int, help = "The number of grid_points in the x-y plane", default = 12)

parsed_args = vars(parser.parse_args())
rm = 2.9673;

def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simps(x*psi, x)
   print ("Norm = " + str(int_psi_square))
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

R = parsed_args["R"]
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = parsed_args["n"];
ny = parsed_args["n"];
nz = parsed_args["nz"];
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

nrm = dx*dy*np.sqrt(dz*rm)*rm*rm
fname = parsed_args["filename"]
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
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
plt.plot(rval,radial, 'g*')
plt.show()

meanr = 2*np.pi*25*intgr.simps(rval*rval*radial, rval)/norm
spatial_ext = np.sqrt(2*np.pi*25*intgr.simps(rval*((rval - meanr)**2)*radial, rval)/norm)
print('Spatial extent = ' + str(spatial_ext))
print('Mean r = ' + str(meanr))
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
