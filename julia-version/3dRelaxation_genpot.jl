import Elliptic
import PyPlot
import FFTW
using LazyGrids: ndgrid
using NPZ

rm = 2.9673;

#Compute Recoil Energy
hbar = 11606  * 6.5821 * 10^(-16) #Kelvin second
hbar2by2 = hbar^2/2 #Kelvin^2 second^2
mass = 4*9.3827*11606 * (10 ^ 8) #Kelvin/c^2
c = 2.998 * 10^(18) #Angstrom per second
E_rec = hbar2by2/(mass*(rm^2))*(c^2)
println(E_rec)

function F(x,D)
    a = zeros(size(x))
    for ix in CartesianIndices(x)
        if (x[ix] < D)
            a[ix]  = exp(-(D/x[ix] - 1.0)*(D/x[ix] - 1.0))
        else
            a[ix] = 1.0
        end
    end
    return a
end

function Aziz_potential(r)
    A       = 0.5448504E6;
    epsilon = 10.8;
    alpha   = 13.353384
    D       = 1.241314;
    C6      = 1.3732412;
    C8      = 0.4253785;
    C10     = 0.1781;

    x = r

    Urep = A * exp.(-alpha*x);

    ix2 = 1.0 ./ (x .* x);
    ix6 = ix2 .* ix2 .* ix2;
    ix8 = ix6 .* ix2;
    ix10 = ix8 .* ix2;
    Uatt = -( C6*ix6 .+ C8*ix8 .+ C10*ix10 ) .* F(x,D);
    return ( epsilon * (Urep .+ Uatt) );
end

function PIMC_potential(r,R)
    epsilon = 1.4212942968311457
    sigma = 5.442152204855649/rm
    density = 0.0087*(rm^3)
    x = r ./ R;
    x[x .> 1] .= 1 
    
    x2 = x.*x;
    x4 = x2.*x2;
    x6 = x2.*x4;
    x8 = x4.*x4;
    f1 = 1.0 ./ (1.0 .- x2);
    sigoR3 = (sigma/R) ^ 3
    sigoR9 = sigoR3*sigoR3*sigoR3;
    v9 = (1.0.*(f1.^9.0)/(240.0)) .* ((1091.0 .+ 11156 * x2 .+ 16434*x4 .+ 4052*x6 .+ 35*x8).*Elliptic.E.(x2) .- 8.0.*(1.0 .- x2).*(1.0 .+ 7*x2).*(97.0 .+ 134*x2 .+ 25*x4).*Elliptic.K.(x2));
    v3 = 2.0.*(f1.^3.0) .* ((7.0 .+ x2).*Elliptic.E.(x2) .- 4.0*(1.0.-x2).*Elliptic.K.(x2));
    val = (pi*epsilon*sigma*sigma*sigma*density/3.0).*(sigoR9*v9 .- sigoR3*v3)
        
    return val
end

function W(x1,x2,y1,y2,zr)
    #Kelvin
    R = 8.65/rm
    rho1 = sqrt.(x1.^2 + y1.^2)
    rho2 = sqrt.(x2.^2 + y2.^2)
    r = sqrt.((x1 .- x2).^2 .+ (y1 .- y2).^2 .+ zr.^2)
    piece1 = PIMC_potential(rho1,R)
    piece1[isnan.(piece1)] .= 500
    #print(piece1)
    #PyPlot.pcolormesh(piece1)
    #PyPlot.colorbar()
    #PyPlot.show()
    piece2 = PIMC_potential(rho2,R)
    piece2[isnan.(piece2)] .= 500
    #PyPlot.pcolormesh(piece2)
    #PyPlot.colorbar()
    #PyPlot.show()
    piece3 = Aziz_potential(r)
    piece3[isnan.(piece3)] .= 500
    #PyPlot.pcolormesh(piece3)
    #PyPlot.colorbar()
    #PyPlot.show()
    return (piece1 .+ piece2 .+ piece3)./E_rec
end


#Cartesian Space
Lx = 2*8.65; Ly = 2*8.65; Lz = 25;
Lxnd = Lx/rm; 
Lynd = Ly/rm;
Lznd = Lz/rm;

nx = 72; ny = 72; nz = 12;
dx = Lxnd/nx; dy = Lynd/ny; dz = Lznd/nz;
x = dx * (-nx/2:nx/2-1)
y = dy * (-ny/2:ny/2-1)
z = dz * (-nz/2:nz/2-1)
(x1,x2,y1,y2,zr) = ndgrid(x,x,y,y,z)

coff = 300/E_rec
Wcut = W(x1,x2,y1,y2,zr)
Wcut[Wcut .> coff] .= coff
#PyPlot.pcolormesh(Wcut[:,36,:,36,1])
#PyPlot.colorbar()
#PyPlot.show()

npzwrite("3dPot-test.npy",Wcut)
print("Saved to file")
