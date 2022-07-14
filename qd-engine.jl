#! /usr/bin/env -S julia --project=/work/qd-engine/project_1dqd

println("Loading packages...")
#using Threads
using LinearAlgebra, FFTW
using Statistics
using JSON
using Printf
using SpecialPolynomials


#===================================================
                Basic constants
===================================================#

wn_to_au = 5.29177210903e-9
c_au = 137.0359991				# Speed of light in a.u.
c = 29979245800                 # Speed of light in cm/s
fs_to_au = 41.341373335
ang_to_bohr = 1.889726125
μ_to_me = 1822.88849            # Atomic mass unit in masses of an electron
e = 1.60218e-19                 # e⁻ charge
h = 6.62607e-34                 # Planck constant
kB = 3.16681e-6                 # Boltzmann constant in a.u.


#===================================================
                Function section
===================================================#

function read_potential(filepath::String)
    #= Potential is given as two columns:
        position | potential energy =#
    open(filepath) do file
        x_axis = Array{Float64}(undef,0)
        potential = Array{Float64}(undef,0)
        for line in eachline(file)
            if startswith("#", line)
                continue
            end
            line = parse.(Float64, split(line))
            append!(x_axis, line[1])
            append!(potential, line[2])
        end
        return (x_axis, potential)
    end
end



function propagate(potential::Array{Float64}, p::Float64,
        x_space::Array, dt::Float64, wf::Array)
    #= Function for a propagation of a wavepacket in one dimension
        using the Direct Fourier Method.
        All input variables are in atomic units.
        potential => Array of values of potential energy
        p => constant for propagation
        x_space => Array of spatial coordinates
        dt => time increment 
        wf => wavefunction at time t (Complex array)
        NOTE: ADD PREPARED FFT=#
    
    N = length(x_space)
    i_k = -N/2:N/2-1        # Assuming symmetric x range
    wf = fft(wf)
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*p*dt/2) * i_k.^2 )
    wf = fftshift(wf)
    wf = ifft(wf)
    wf = wf .* exp.( -(im*potential*dt) )
    wf = fft(wf)
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*p*dt/2) * i_k.^2 )
    wf = fftshift(wf)
    wf = ifft(wf)
    return wf 
end

function save_data(data::Array, x_space::Array, dt::Float64)
    open("PAmp.dat", "w") do file
        header = "# time [fs] Probability amplitude [Bohrs]---->\n"
        write(file, header)
        write(file, " "^12)
        line = map(x -> @sprintf("%12.4f", x), x_space)
        for field in line
            write(file, field)
        end
        write(file, "\n")
        for (i, dat) in enumerate(eachrow(data))
            time = @sprintf "%12.4f" i*stride*dt/fs_to_au
            write(file, time)
            dat = Float64.( conj(dat) .* dat )
            line = map(x -> @sprintf("%12.7f", x), dat)
            for field in line
                write(file, field)
            end
            write(file, "\n")
        end
    end
    open("PAmp-kspace.dat", "w") do file
        header = "# time [fs] Probability amplitude [momentum]---->\n"
        write(file, header)
        write(file, " "^12)
        map( x -> write(file, @sprintf("%12d", x)), -length(x_space)/2:(length(x_space)/2-1) )
        write(file, "\n")
        for (i, dat) in enumerate(eachrow(data))
            if mod(i,2) == 0
                continue
            end
            time = @sprintf "%12.4f" i*stride*dt/fs_to_au
            write(file, time)
            dat = Array{ComplexF64}(dat)
            k_dat = fftshift(fft(dat))
            k_dat = Float64.( conj(k_dat) .* k_dat )
            map(x -> write(file, @sprintf("%12.7f", x)), k_dat)
            write(file, "\n")
        end
    end
end

function save_vec(cf::Array, filename::String, dt::Float64)
    open(filename, "w") do file
        header = "# time [fs] val\n"
        write(file, header)
        for (i, val) in enumerate(cf)
            time = @sprintf "%12.4f" i*stride*dt/fs_to_au
            write(file, time)
            val = @sprintf "%12.7f\n" val
            write(file, val)
        end
    end
end

function compute_spectrum(cf::Array, timestep::Number)
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*10) ]
    spectrum = abs.(fft(cf)) .^ 2
    spectrum = spectrum/norm(spectrum)
    dv = 1/(length(cf)*timestep*1e-15)
    c = 29979245800
    wns = [ i*dv/c for i in 0:length(cf)-1 ]
    spectrum = wns .* spectrum
     
    open("pos-spectrum.dat", "w") do file
        for (wn, val) in zip(wns, spectrum)
            if wn > 4000
                break
            end
            line = @sprintf "%12.7f%12.7f\n" wn val
            write(file, line)
        end
    end
end

function electronic_spectrum(cf::Array, timestep::Number, zpe::Number, name::String)
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*10) ]
    spectrum = real.(fft(cf))
    spectrum = spectrum/norm(spectrum)
    dv = 1/(length(cf)*timestep*1e-15)
    c = 29979245800
    wns = [ (i*dv + zpe)/c for i in 0:length(cf)-1 ]
    spectrum = wns .* spectrum

    open("$name.dat", "w") do file
        for (wn, val) in zip(wns, spectrum)
            if wn > 20000
                break
            end
            line = @sprintf "%12.7f%12.7f\n" wn val
            write(file, line)
        end
    end
end


function tr_re(wf::Array, x_space::Array, delim::Float64, T::Number, 
        dt::Float64, stride::Int)
    
    dx = (x_space[end] - x_space[1])/(length(x_space)-1)
    delim_i = Int(round((delim - x_space[1])/dx))
    T_i = Int(round(T/dt/stride))
    tr_ref = wf[T_i, delim_i+1:end]
    re_ref = wf[T_i, 1:delim_i]
    cf_len = length(eachrow(wf)) - T_i + 1
    cf_re = Array{Float64}(undef, cf_len)
    cf_tr = Array{Float64}(undef, cf_len)
    cf_wf = Array{Float64}(undef, cf_len)
    # Function for <ψ(T)|ψ(T+t)>
    cf_t(wf_t::Array, wf_ref::Array) = abs(dot( wf_ref, wf_t))
    # Compute the correlation function with time T as reference:
    for (i,t) in enumerate(eachrow(wf[T_i:end,:]))
        tr = t[delim_i+1:end]
        re = t[1:delim_i]
        (re_ovl, tr_ovl, wf_ovl) = [ cf_t(re, re_ref), cf_t(tr, tr_ref), cf_t( t[:], wf[T_i,:]) ]
        cf_re[i] = re_ovl
        cf_tr[i] = tr_ovl
        cf_wf[i] = wf_ovl
    end
    # Save results:
    open("reflection-transmission_CFs.dat", "w") do file
        head = @sprintf "#%13s%14s%14s%14s\n" "time [fs]" "total" "reflected" "transmitted"
        write(file, head)
        for itime in 1:cf_len
            time = (itime-1)*stride*dt/fs_to_au
            line = @sprintf "%14.4f%14.7f%14.7f%14.7f\n" time cf_wf[itime] cf_re[itime] cf_tr[itime]
            write(file, line)
        end
    end
    # Compute transmission and reflection coefficients
    do_spec(cf::Array) = abs.(fft( [ cf .- mean(cf); zeros(length(cf)*3) ] ))
    σTr = do_spec(cf_tr) 
    σRe = do_spec(cf_re)
    σTot = σTr + σRe
    re_E = σRe ./ σTot
    tr_E = σTr ./ σTot
    dE = 1/(length(cf)*stride*dt/fs_to_au*1e-15)*h/e # dE in eV
    engs = [ i*dE for i in 0:length(σTot)-1 ]
    open("reflection-transmission_Coeffs.dat", "w") do file
        head = @sprintf "#%13s%14s%14s%14s%14s%14s\n" "Energy [eV]" "σTot" "σRe" "σTr" "Re" "Tr"
        write(file, head)
        for i in 1:length(engs)
            if engs[i] > 5
                break
            end
            line = @sprintf "%14.4f%14.7f%14.7f%14.7f%14.7f%14.7f\n" engs[i] σTot[i] σRe[i] σTr[i] re_E[i] tr_E[i]
            write(file, line)
        end
    end
end

function construct_HarmWP(temp::Number, x0::Float64, k::Float64, μ::Float64,
        x_space::Array{Float64})
    
    chi(n::Int, x::Float64) = 1/(sqrt(2^n*factorial(n))) * basis(Hermite, n)((μ*k)^(1/4)*(x-x0)) * exp( -(1/2*sqrt(μ*k) * (x-x0)^2) )
    energy(n::Int) = sqrt(k/μ) * (n+1/2)
    Z(temp::Number) = 1/(1 - exp( -(sqrt(k/μ)/kB/temp) ) )
    wf0 = zeros(length(x_space))
    for n in 1:10
        pop = exp( -( (energy(n-1) - energy(0))/kB/temp ) )/Z(temp)
        wfn = map( x -> -1*pop*chi(n-1, x), x_space)
        wf0 += wfn
    end
    wf0 = wf0/sqrt(dot(wf0, wf0)) 
    return wf0
end

#===================================================
                Prepare dynamics 
===================================================#

# Read input:
input = JSON.parsefile("input.json")

# Essential parameters
(x_space, potential) = read_potential(input["potential"])
μ = input["mass"] * μ_to_me
k_harm = input["k_harm"]
x0 = input["initpos"]
wf0 = exp.( -(1/2*sqrt(μ*k_harm) * (x_space .- x0).^2 ))
wf0 = wf0/sqrt(dot(wf0,wf0))
p = (1/2/μ)*(1/(x_space[end] - x_space[1]))^2        # Constant for kinetic energy propagation
dt = input["dt"]
stride = input["stride"]
tmax = input["tmax"]

# Construct initial WP as a superposition of harmonic states weighted by Boltzmann factors
if haskey(input, "boltzmann")
    temp = input["boltzmann"]
    wf0 = construct_HarmWP(temp, x0, k_harm, μ, x_space)
end

# Read dipole moment (if available)
if haskey(input, "dipole")
    (space, dipole) = read_potential(input["dipole"])
end

# Calculation of electronic spectra (NOT WORKING PROPERLY)
haskey(input, "vibres_spec") ? vibres = true : vibres = false

# Prepare calculation of reflection and transmission coeffs.:
if haskey(input, "scatter")
    top_bar = input["top_bar"]
    scatter_T = input["scatter_T"]
end

# Initialize output:
N_records = Int(fld(tmax/dt, stride))   # Number of records in output files
data = Array{Complex}(undef, N_records, length(x_space))    # Ψ(t)
cf = Array{ComplexF64}(undef, N_records)       # correlation function <ψ(0)|ψ(t)>
dip_cf = Array{Float64}(undef, N_records)   # dipole corr.func. <ψ(0)|μ|ψ(t)>


#===================================================
                Execute dynamics
===================================================#

println("Running dynamics...")
istep = 1
t = 0
wf = wf0 

while t < tmax
    global wf
    global t, istep
    
    # Propagate the wavepacket:
    wf  = propagate(potential, p, x_space, dt, wf)
    t += dt
    
    # Save data:
    if mod(istep, stride) == 0
        i_rec = Int(istep/stride)
        data[i_rec, :] = wf 
        cf[i_rec] = abs(dot(wf0, wf))
        if haskey(input, "dipole")
            dip_cf[i_rec] = Float64(dot(wf0, dipole .* wf))
        end
    end
    istep += 1
end

#===================================================
            Analyze and save the results
===================================================#

println("Analyzing and saving...")
save_data(data, x_space, dt)            # Save |Ψ(x,t)|² and |Ψ(k,t)|²
save_vec(cf, "corrF.dat", dt )          # Save |<ψ(0)|ψ(t)>|
compute_spectrum(cf, dt*stride/fs_to_au)
#compute_spectrum(dip_cf, dt*stride/fs_to_au, "dip-spectrum")

# Compute reflection and transmission coefficients
if haskey(input, "scatter")
    tr_re(data, x_space, top_bar, scatter_T, dt, stride)
end

# Compute electron absorption spectrum
if vibres
    zpe = 1/2/pi*sqrt(k_harm/μ) * fs_to_au * 1e15
    electronic_spectrum(cf, dt*stride/fs_to_au, zpe , "vibres-spectrum")
end
