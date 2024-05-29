import Elliptic
import PyPlot
import FFTW

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

function W(x1,x2)
    #Kelvin
    R = 8.65/rm
    piece1 = PIMC_potential(abs.(x1),R)
    piece1[isnan.(piece1)] .= 500
    #print(piece1)
    #PyPlot.pcolormesh(piece1)
    #PyPlot.colorbar()
    #PyPlot.show()
    piece2 = PIMC_potential(abs.(x2),R)
    piece2[isnan.(piece2)] .= 500
    #PyPlot.pcolormesh(piece2)
    #PyPlot.colorbar()
    #PyPlot.show()
    piece3 = Aziz_potential(abs.(x1-x2))
    piece3[isnan.(piece3)] .= 500
    #PyPlot.pcolormesh(piece3)
    #PyPlot.colorbar()
    #PyPlot.show()
    return (piece1 .+ piece2 .+ piece3)./E_rec
end


#Cartesian Space
Lx = 25;
Lxnd = Lx/rm;

nx = 1000;
dx = Lxnd/nx
x = dx * (-nx/2:nx/2-1)
x2 = x' .* ones(size(x))
x1 = ones(size(x))' .* x

#Fourier Space
dkx = 2*pi/Lxnd;
kx = dkx * (vcat(0:nx/2-1,-nx/2:-1))
kx2 = kx' .* ones(size(kx))
kx1 = ones(size(kx))' .* kx

#Laplacian in Fourier Space
fftminlap = kx1.^2 + kx2.^2

coff = 300/E_rec
Wcut = W(x1,x2)
Wcut[Wcut .> coff] .= coff
"""
PyPlot.pcolormesh(Wcut)
PyPlot.colorbar()
PyPlot.show()
"""

#Preconditioning-operator
fftPinv = 1 ./(coff .+ fftminlap)

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
dt = 1.0
s = 0.5
center = 1.0
width = 0.6
psi = exp.((-(x1 .+ center).^2 .- (x2 .- center).^2)./(width^2))
psi = sqrt(2).*psi./sqrt(sum(psi.^2)*dx*dx)
A = sqrt(2 ./(4 .+ 2*sum(conj.(psi) .* psi')*dx*dx))
psi = A*(psi + psi')
#PyPlot.pcolormesh(psi)
#PyPlot.colorbar()
#PyPlot.show()
println(sum(psi.^2)*dx*dx)

error = 1
erroreqs = 1
i = 0
L0psinold = 0
E2 = 0
"""
#Plot 1st particle
p = vec(sum(psi.^2, dims = 1)*dx)
PyPlot.plot(x,p)


#Plot 2nd particle
p = vec(sum(psi.^2, dims = 2)*dx)
PyPlot.plot(x,p)
PyPlot.show()
"""
rec_error = []
eqs_error = []
alpha_err = []

while erroreqs > 1e-5
    global i = i + 1

    if i != 1
        #L0psislow = L0psin - L0psinold
        global L0psinold = L0psin
        global psi = deepcopy(psi_n)
    end

    potentialpart = Wcut .* psi
    L00psin = real(FFTW.ifft(fftminlap .* FFTW.fft(psi))) .+ potentialpart
    num = sum(Pinv(psi) .* L00psin)
    
    den = sum(Pinv(psi) .* psi)
    
    E1 = num/den
    global L0psin = L00psin - E1*psi
    if (i > 100)
        L0psislow = L0psin - L0psinold
        alpha = sum(psislow .* L0psislow) ./ sum(psislow .* P(psislow))
        push!(alpha_err, alpha)
        gamma = min(1 - s/(alpha*dt),0)        
        subtracted_mode = gamma*psislow*((sum(psislow .* L0psin)))/(sum(psislow .* P(psislow)))      
    else
        subtracted_mode = zeros(size(psi))
    end
    global psi_n = psi .- dt*Pinv(L0psin) .+ subtracted_mode*dt
    #psi_n = (1/np.sqrt(2))*(psi_n + np.conj(psi_n).T)
    psi_n = sqrt(2) * psi_n ./ sqrt(sum(psi_n.^2)*dx*dx)
    #print(np.sum(psi_n**2)*dx*dx)
    #Calculate normalisation
    #A = np.sqrt(2/(4 + 2*np.sum(np.multiply(np.conj(psi_n),psi_n.T))*dx*dx))
    #psi_n = A*(psi_n + psi_n.T)
    #print(np.sum(psi**2)*dx*dx)
    global psislow = psi_n .- psi

    global error = real(sqrt(sum(psi_n .- psi).^2))
    push!(rec_error, real(error))
    PyPlot.semilogy(rec_error,"r+")
    global E2 = sum(psi .* L00psin) / (sum(psi .* psi))
    global erroreqs = real(sqrt(sum((L00psin .-  E2*psi).^2)))
    push!(eqs_error, real(erroreqs))
    PyPlot.semilogy(eqs_error,"b*")

    if false
        PyPlot.show()
    end
end

println(E2)
println(i)
println(error)
#PyPlot.show()
"""
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
"""
