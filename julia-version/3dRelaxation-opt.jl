#using Pkg
#Pkg.activate(".")
#Pkg.add(["PyPlot","FFTW","LazyGrids","NPZ","MKL"])

using MKL
import PyPlot
import FFTW
using LazyGrids: ndgrid, ndgrid_array
using NPZ

const rm = 2.9673;

#Compute Recoil Energy
const hbar = 11606  * 6.5821 * 10^(-16) #Kelvin second
const hbar2by2 = hbar^2/2 #Kelvin^2 second^2
const mass = 4*9.3827*11606 * (10 ^ 8) #Kelvin/c^2
const c = 2.998 * 10^(18) #Angstrom per second
const E_rec = hbar2by2/(mass*(rm^2))*(c^2)

#Cartesian Space
const Lx = 2*8.65; 
const Ly = 2*8.65; 
const Lz = 25;
const Lxnd = Lx/rm;
const Lynd = Ly/rm;
const Lznd = Lz/rm;

const nx = 72; const ny = 72; const nz = 12;
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

const coff = 300/E_rec

"""
PyPlot.pcolormesh(Wcut)
PyPlot.colorbar()
PyPlot.show()
"""

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

#Iteration
function Iterations(Wcut)
    dt::Float64 = 1.2
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

        if false
            PyPlot.show()
        end
    end
    return E2, i, error, psi
end

function main()
    println(E_rec)
    Wcut = npzread("3dPot-test.npy")
    E2, i, error, psi = Iterations(Wcut)
    println(E2)
    println(i)
    println(error)
    npzwrite("Final-wavefunction.npy",psi)
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

