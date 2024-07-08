#using Pkg
#Pkg.activate(".")
#Pkg.add(["Elliptic","PyPlot","FFTW","LazyGrids","NPZ","MKL","ArgParse"])

import Elliptic
using MKL
import PyPlot
import FFTW
using LazyGrids: ndgrid, ndgrid_array
using NPZ
using ArgParse

const rm = 2.9673;

#Parse Command-line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "-R"
        help = "Radius of the nanopore"
        arg_type = Float64
        required = true
    "--Lz"
        help = "Length in the z-direction (in Angstroms)"
        arg_type = Float64
        default = 25
    "--epsilon","-e"
        help = "Epsilon for the Leonard-Jones interaction"
        arg_type = Float64
        default = 1.4212942968311457
    "--sigma","-s"
        help = "Sigma for the Lennard-Hones interaction (in Angstroms)"
        arg_type = Float64
        default = 5.442152204855649
    "--rho"
        help = "Density of the surface (in Length inverse cubed)"
        arg_type = Float64
        default = 0.0087
    "-n"
        help = "Number of points in the x and y direction"
        arg_type = Int
        default = 72
    "--nz"
        help = "Number of points in the z-direction"
        arg_type = Int
        default = 12
    "--cutoff","-c"
        help = "Potential cutoff (in Kelvin)"
        arg_type = Float64
        default = 300
    "--filename","-f"
        help = "Filename ending"
        arg_type = String
        default = "-jl"
    "--debug"
        help = "A flag to activate the debug loop"
        action = :store_true
end

parsed_args = parse_args(ARGS, s)

#Compute Recoil Energy
const hbar = 11606  * 6.5821 * 10^(-16) #Kelvin second
const hbar2by2 = hbar^2/2 #Kelvin^2 second^2
const mass = 4*9.3827*11606 * (10 ^ 8) #Kelvin/c^2
const c = 2.998 * 10^(18) #Angstrom per second
const E_rec = hbar2by2/(mass*(rm^2))*(c^2)

#Cartesian Space
const R = parsed_args["R"]
const Lx = 2*R; 
const Ly = 2*R; 
const Lz = parsed_args["Lz"]
const Lxnd = Lx/rm;
const Lynd = Ly/rm;
const Lznd = Lz/rm;

const nx = parsed_args["n"]; const ny = parsed_args["n"]; const nz = parsed_args["nz"];
const dx = Lxnd/nx; const dy = Lynd/ny; const dz = Lznd/nz;
const x = dx * (-nx/2:nx/2-1)
const y = dy * (-ny/2:ny/2-1)
const z = dz * (-nz/2:nz/2-1)
(x1,x2,y1,y2,zr) = ndgrid(x,x,y,y,z)

#Fourier Space
const dkx = 2*pi/Lxnd; const dky = 2*pi/Lynd; const dkz = 2*pi/Lznd;
const kx = dkx * (vcat(0:nx/2-1,-nx/2:-1))
const ky = dky * (vcat(0:ny/2-1,-ny/2:-1))
const kz = dkz * (vcat(0:nz/2-1,-nz/2:-1))

(kx1, kx2, ky1, ky2, kzr) = ndgrid(kx,kx,ky,ky,kz)
#Laplacian in Fourier Space
const fftminlap = kx1.^2 + kx2.^2 + ky1.^2 + ky2.^2 + 2*kzr.^2

#Defines the cutoff
const coff = parsed_args["cutoff"]/E_rec

#Preconditioning-operator
const fftPinv = 1 ./(coff .+ fftminlap)

#Define Pinv of a function
function Pinv(f)
    piece1 = FFTW.fft(f)
    #display(piece1)
    piece2 = fftPinv .*  FFTW.fft(f)
    #display(piece2)
    piece3 = real(FFTW.ifft(piece2))
    #display(piece3)
    return piece3
end

function P(f)
    return coff*f .+ real(FFTW.ifft(fftminlap .* FFTW.fft(f)))
end

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
    epsilon = parsed_args["epsilon"]
    sigma = parsed_args["sigma"]/rm
    density = parsed_args["rho"]*(rm^3)
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
    R = parsed_args["R"]/rm
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

#Iteration
function Iterations(Wcut)
    dt::Float64 = 1.0
    s::Float64 = 0.1
    center::Float64 = 1.0
    width::Float64 = 0.6
    psi = exp.((.-(x1 .+ center).^2 .- (x2 .- center).^2 .- (y1 .+ center).^2 .-(y2 .- center).^2)./(width^2)) 
    
    psi = sqrt(2).*psi./sqrt(sum(psi.^2)*dx*dx*dy*dy*dz)
    #println(sum(psi.^2)*dx*dx*dy*dy*dz)
    A::Float64 = sqrt(2 ./(4 .+ 2*sum(permutedims(conj(psi),(3,4,1,2,5)) .* permutedims(psi,(2,1,4,3,5)))*dx*dx*dy*dy*dz))
    #println(A)
    psi = A*(psi .+ permutedims(psi,(2,1,4,3,5)))
    #PyPlot.pcolormesh(psi)
    #PyPlot.colorbar()
    #PyPlot.show()
    println(sum(psi.^2)*dx*dx*dy*dy*dz)
    
    psi_old = zeros(size(psi))
    error::Float64 = 1
    erroreqs::Float64 = 1
    i::Int = 0
    psi_old = zeros(size(psi))
    psislow = zeros(size(psi))
    L0psinold = zeros(size(psi))
    L0psislow = zeros(size(psi))
    L0psin = zeros(size(psi))
    subtracted_mode = zeros(size(psi))
    E2::Float64 = 0

    rec_error = []
    eqs_error = []
    alpha_err = []

    while erroreqs > 1e-6
        i = i + 1
        
        psi_old .= psi

        L00psin = real(FFTW.ifft(fftminlap .* FFTW.fft(psi))) .+ (Wcut .* psi)

        p_inv_psi = Pinv(psi)
        num = sum(p_inv_psi .* L00psin)
        den = sum(p_inv_psi .* psi)
        #println(den)
        E1::Float64 = num/den
        
        L0psin .= L00psin - E1*psi
        if (i > 1)
            L0psislow = L0psin - L0psinold
        end
        if (i > 100)
            L0psislow = L0psin .- L0psinold
            alpha = sum(psislow .* L0psislow) ./ (sum(psislow .* P(psislow)))
           
            #push!(alpha_err, alpha)
            gamma = min(1 - s/(alpha*dt),0)        
            subtracted_mode = gamma*psislow*((sum(psislow .* L0psin)))/(sum(psislow .* P(psislow)))      
        else
            subtracted_mode = zeros(size(psi))
        end
        psi = psi .- dt*Pinv(L0psin) .+ subtracted_mode*dt
        #psi_n = (1/np.sqrt(2))*(psi_n + np.conj(psi_n).T)
        psi = sqrt(2) * psi ./ sqrt(sum(psi.^2)*dx*dx*dy*dy*dz)
        #print(np.sum(psi_n**2)*dx*dx)
        #Calculate normalisation
        Apr::Float64 = sqrt(2 ./(4 .+ 2*sum(permutedims(conj(psi),(3,4,1,2,5)) .* permutedims(psi,(2,1,4,3,5)))*dx*dx*dy*dy*dz))
        psi .= Apr*(psi .+ permutedims(psi,(2,1,4,3,5)))
   
        #print(np.sum(psi**2)*dx*dx)
        psislow .= psi .- psi_old
        L0psinold .= L0psin

        error = real(sqrt(sum(psi .- psi_old).^2))
        push!(rec_error, real(error))
        PyPlot.semilogy(rec_error,"r+")
        E2 = sum(psi .* L00psin) / (sum(psi .* psi))
        erroreqs = real(sqrt(sum((L00psin .-  E2*psi).^2)))
        push!(eqs_error, real(erroreqs))
        PyPlot.semilogy(eqs_error,"b*")

        if parsed_args["debug"] && i % 100 == 0 
            PyPlot.show()
        end
    end
    return E2, i, error, psi
end

function main()
    println(E_rec)
    Wcut = W(x1,x2,y1,y2,zr)
    Wcut[Wcut .> coff] .= coff
    println("Created potential")
    E2, i, error, psi = Iterations(Wcut)
    print(sum(psi.^2)*dx*dx*dy*dy*dz)
    println(E2)
    println(i)
    println(error)
    fend = parsed_args["filename"]
    fname = "Final-wavefunction" * fend * ".npy"
    npzwrite(fname,psi)
end

#PyPlot.show()
#=
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
PyPlot.pcolormesh(real(psi_n))
PyPlot.show()

#Plot 1st particle
p = vec(sum(psi_n.^2, dims = 1)*dx)
PyPlot.plot(x,p)

#Plot 2nd particle
p = vec(sum(psi_n.^2, dims = 2)*dx)
PyPlot.plot(x,p)
PyPlot.show()
=#

main()

