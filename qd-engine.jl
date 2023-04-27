#! /usr/bin/env -S julia --project=/work/qd-engine/project_1dqd

println("Loading packages...")
#using Threads
using LinearAlgebra, FFTW
using Statistics
using YAML
using Printf
using SpecialPolynomials
using UnicodePlots

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
                    INFO section
===================================================#
function print_hello()
    hello = """

\t#====================================================
\t     One-dimensional Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

   """
   println(hello)
end

function print_init()
    ω = sqrt(k_harm/μ)
    ν = ω/2/pi * fs_to_au*1e15/c
    println("\n    Basic information:\n")
    println("""\tUsing Direct Fourier Method for propagation, see:
            \t  Kosloff & Kosloff J. Chem. Phys., 79(4), 1983.\n
            \tfor the split operator formalism see:
            \t  Feit, Fleck & Steiger, J. Comput. Phys., 1982, 47, 412–433.\n""")
    println(@sprintf "\t Particle mass:%8.4f amu" μ/μ_to_me )
    println("\t Using Gaussian WP at t=0:")
    println(@sprintf "\t\t%-20s%12.3f Bohrs" "centered at:" x0 )
    println(@sprintf "\t\t%-20s%12.3f cm⁻¹" "harm. freq.:" ν )
    println(@sprintf "\t\t%-20s%12.3f Bohrs\n" "FWHM:" 2*sqrt(2*log(2))/(k_harm*μ)^(1/4) )
    println(@sprintf "\t%-20s%12.2f a.u." "Timestep:" dt )
    println(@sprintf "\t%-20s%12.f a.u." "tₘₐₓ:" tmax )
    println(@sprintf "\t%-20s%12d" "stride:" stride )
    println(@sprintf "\t%-20s%12.2e Bohrs" "Grid spacing:" dx )
    println(@sprintf "\t%-20s%12d" "Grid points:" length(potential) )
    println(@sprintf "\t%-20s%12.4f a.u." "Maximal momentum:" 1/2/dx )
    println(@sprintf "\n\t%-22s%12s\n" "Potential taken from:" input["potential"] )
       
    println("\t" * "*"^60)
    println("\tInitial condition:\n")
    
    plt_min = Int(round(0.1*length(x_space)))
    plt_max = Int(round(0.9*length(x_space)))
    pot_max = max(potential[plt_min:plt_max]...)
    pot_min = min(potential[plt_min:plt_max]...)
    wf_max = max(wf0...)
    initPlot = lineplot(x_space[plt_min:plt_max], potential[plt_min:plt_max], 
                       canvas=BrailleCanvas,
                       border=:ascii, color=:white, name="potential",
                       xlabel="Distance [Bohrs]", ylabel="Energy [Hartree]",
                       height=10, grid=false, compact=true, blend=false)
    wf_rescale = 0.5*(pot_max - pot_min)/wf_max
    wf_to_plot = wf0 * wf_rescale .+ 1.1*pot_min
    lineplot!(initPlot, x_space, wf_to_plot, color=:cyan, name="WF(t=0)") 
    println(initPlot)
    println("\t" * "*"^60)
end

function print_run()
    message = """

\t#====================================================
\t              Running dynamics:
\t====================================================#

"""
    println(message)
end

function print_analyses()
    message = """

\t#====================================================
\t              Analyzing and saving:
\t====================================================#

\tWP propability amplitudes written to "PAmp.dat"
\t  in momentum representation written to "PAmp-kspace.dat"

\tAnalyses requested:
"""
    println(message)
    println("\t" * "*"^60)
    println(@sprintf "\t\t%-22s%20s" "Type" "Destination file")
    println("\t" * "*"^60)
    println(@sprintf "\t\t%-22s%20s" "pos-pos corr. func." "corrF.dat")
    if haskey(input, "vibronic")
        println(@sprintf "\t\t%-22s%20s" "vibronic spectrum:" "spectrum.dat")
    elseif haskey(input, "resonanceRaman")                                     
        println(@sprintf "\t\t%-22s%20s" "resonance Raman:" "resRaman.dat")    
        println(@sprintf "\t\t%-22s%20s" "energy levels:" "spectrum.dat")
        println(@sprintf "\t\t%-22s%20s" "0→ 1 Raman overlap:" "Raman-CF.dat")
   else
        println(@sprintf "\t\t%-22s%20s" "energy levels:" "spectrum.dat")
    end
    if haskey(input, "scatter")
        println(@sprintf "\t\t%-22s" "Scattering analysis")
        println(@sprintf "\t\t%42s" "reflection-transmission_CFs.dat")
        println(@sprintf "\t\t%42s" "reflection-transmission_coeffs.dat")
    end
end

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
        x_space::Array, dt::Number, wf::Array)
    #= Function for a propagation of a wavepacket in one dimension
        using the Direct Fourier Method.
        All input variables are in atomic units.
        potential => Array of values of potential energy
        p => constant for propagation
        x_space => Array of spatial coordinates
        dt => time increment 
        wf => wavefunction at time t (Complex array)
        TODO: ADD PREPARED FFT
              Change order of exp(-i/h T) and exp(-i/h V) =#
    
    N = length(x_space)
    i_k = -N/2:N/2-1
    wf = wf .* exp.( -(im*potential/2*dt) )
    wf = fft(wf)
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*p*dt) * i_k.^2 )
    wf = fftshift(wf)
    wf = ifft(wf)
    wf = wf .* exp.( -(im*potential/2*dt) )
#    wf = fft(wf)
#    wf = fftshift(wf)
#    wf = wf .* exp.( -(im*p*dt/2) * i_k.^2 )
#    wf = fftshift(wf)
#    wf = ifft(wf)
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
    #= Save an array of numbers for each timestep =#
    open(filename, "w") do file
        header = "# time [fs] val\n"
        write(file, header)
        for (i, val) in enumerate(cf)
            time = @sprintf "%12.4f" (i-1)*stride*dt/fs_to_au
            write(file, time)
            val = @sprintf "%12.7f\n" val
            write(file, val)
        end
    end
end

function compute_spectrum(cf::Array, timestep::Number; 
        maxWn::Number=4000, zpe::Number=0, name::String="spectrum")
    #= Compute spectrum from auto-correlation function.
        cf => Complex array
        timestep => dt*stride in fs 
        upperBound => save spectrum up to certain wavenumber 
        TODO: add lineshape function =#
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*5) ]
    spectrum = abs.(fft(cf)) .^ 2
    dv = 1/(length(cf)*timestep*1e-15)
    wns = [ zpe + i*dv/c for i in 0:length(cf)-1 ]
    spectrum = wns .* spectrum
    spectrum = spectrum/norm(spectrum)
    nyquist = 1/(timestep*1e-15)/2/c
     
    open("$name.dat", "w") do file
        head = @sprintf "# Nyquist freq. = %12.5e [cm⁻¹]\n#%11s%12s\n" nyquist "wn. [cm⁻¹]" "Amplitude"
        write(file, head)
        for (wn, val) in zip(wns, spectrum)
            if wn > maxWn
                break
            end
            line = @sprintf "%12.7f%12.7f\n" wn val
            write(file, line)
        end
    end
end

function tr_re(wf::Array, x_space::Array, delim::Float64, T::Number, 
        dt::Float64, stride::Int)
    #=  Compute the transmission and reflection coeffs. as a function of energy.
        Space has to be divided by the top of the barrier and time after a scattering event needs to be specified.
        Whole procedure is describe in Tannor (2007) Introduction to Quantum Mechanics: A time-dependent perspective; Section 7.1.1 
        OBSOLETE FUNTION =#
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

function imTime_propagation(wf0::Array, gs_potential::Array{Float64}, dt::Number=1.0, tmax::Number=10000)
    #= Function will propagate an arbitrary WF in imaginary time.
        Use to determine the initial WP. =#
    t = 0
    wf = wf0

    while t < tmax
        wf = propagate(gs_potential, p, x_space, -dt*im, wf)
        wf = wf/sqrt(dot(wf,wf))    # Propagation in imag. time is not norm-conserving
        t += dt
    end
    open("init-WF.dat", "w") do file
        head = @sprintf "#%15s%16s%16s%16s\n" "x [Bohr]" "real" "imag" "P.Amp."
        write(file, head)
        pAmp = conj(wf) .* wf
        for i in 1:length(wf)
            line = @sprintf "%16.5e%16.5e%16.7e%16.5e\n" x_space[i] real(wf[i]) imag(wf[i]) pAmp[i]
            write(file, line)
        end
    end
    return wf
end

function compute_energy(wf::Array, potential::Array{Float64}, p::Float64)
    #= Evaluate < ψ | H | ψ > using the Direct Fourier method 
        Requires wavefunction, potential and propagation constant as an input.=#
    N = length(potential)
    i_k = -N/2:N/2-1
    ket = potential .* wf
    ket = fft(ket)
    ket = fftshift(ket)
    ket = p * i_k.^2 .* ket
    ket = fftshift(ket)
    ket = ifft(ket)
    energy = dot(wf,ket)/dot(wf,wf)
    return energy
end

function create_harm_state(n::Int, x_space::Array{Float64}, x0::Number,
        k::Number, μ::Number)
    #= Function returns the harmonic vibrational level.
     Force constant k, mass μ and centre x0 have to be specified. =#
    chi(n::Int, x::Float64) = 1/(sqrt(2^n*factorial(n))) * 
        basis(Hermite, n)((μ*k)^(1/4)*(x-x0)) * 
        exp( -(1/2*sqrt(μ*k) * (x-x0)^2) )
    wf = [ chi(n, i) for i in x_space ]
    wf = wf / sqrt(dot(wf, wf))
    return wf
end

function get_eigenfunctions(wf::Array, x_space::Array{Float64}, energies::Array)
    #= Function compute and save vibrational eigenstates.
        Perform FT of ψ(x,t) along time axis.
        Only certain energies will be computed.
        Energies are expected in cm-1 
        TODO: add to interface =#

    (N_t, N_x) = size(wf)
    eigenstates = Array{ComplexF64}(undef, length(energies), N_x)
    for (i,energy) in enumerate(energies)
        freq = energy*c*1e-15/fs_to_au
        eigenstate = zeros(N_x)
        for (time,wp) in enumerate(eachrow(wf))
            t = (time-1)*stride*dt
            eigenstate = eigenstate + wp*exp(-im*freq*time)
        end
        eigenstates[i,:] = eigenstate
    end
    # Save FT of the wavepacket
    open("eigenstates.txt", "w") do file
        head = @sprintf "#%13s" " "
        head *= join([ @sprintf "%20.2f%8s" e " " for e in energies])
        head *= "\n"
        write(file, head)
        for (i, x) in enumerate(x_space)
            line = @sprintf "%14.4f" x
            for i_state in 1:length(energies)
                val = eigenstates[i_state, i]
                line *= @sprintf "%14.4f%14.4f" real(val) angle(val)
            end
            line *= "\n"
            write(file, line)
        end
     end
end


#===================================================
                Prepare dynamics 
===================================================#

print_hello()

# Read input:
input = YAML.load_file("input.yml")

# Essential parameters
(x_space, potential) = read_potential(input["potential"])
dx = abs(x_space[2]-x_space[1])                     # Grid spacing
μ = input["mass"] * μ_to_me                         # Mass in a.u.
k_harm = input["initWF"]["k_harm"]                  # Force constant for harmonic ground state
x0 = input["initWF"]["initpos"]                     # Inital centre of the wave-packet
wf0 = create_harm_state(0, x_space, x0, k_harm, μ)  # WF(t=0)
p = (1/2/μ)*( 1/(dx * length(x_space)) )^2          # Constant for kinetic energy propagation
dt = input["params"]["dt"]                  
stride = input["params"]["stride"]
tmax = input["params"]["tmax"]


# Initialize output:
N_records = Int(fld(tmax/dt, stride)) + 1      # Number of records in output files
data = Array{Complex}(undef, N_records, length(x_space))    # Ψ(t)
cf = Array{ComplexF64}(undef, N_records)       # correlation function <ψ(0)|ψ(t)>

# Echo input
print_init()
#===================================================
            Imaginary time propagation
===================================================#

if haskey(input, "imagT")
    println("Imaginary time propagation...")
    (x_space, gs_potential) = read_potential(input["imagT"]["gs_potential"])
    if haskey(input["imagT"], "dt") && haskey(input["imagT"], "tmax")
        dtim = input["imagT"]["dt"]
        tmaxim = input["imagT"]["tmax"]
        wf0 = imTime_propagation(wf0, gs_potential, dtim, tmaxim)
    else
        wf0 = imTime_propagation(wf0, gs_potential)
    end
end


#===================================================
                Execute dynamics
===================================================#

print_run()
istep = 1
t = 0
wf = wf0 

data[1, :] = wf
cf[1] = abs(dot(wf0, wf))

while t < tmax
    global wf, t, istep
    
    # Propagate the wavepacket:
    wf  = propagate(potential, p, x_space, dt, wf)
    t += dt
    
    # Save data:
    if mod(istep, stride) == 0
        i_rec = Int(istep/stride) + 1
        data[i_rec, :] = wf 
        cf[i_rec] = abs(dot(wf0, wf))
    end
    istep += 1
end

#===================================================
            Analyze and save the results
===================================================#

print_analyses()
save_data(data, x_space, dt)            # Save |Ψ(x,t)|² and |Ψ(k,t)|²
save_vec(cf, "corrF.dat", dt )          # Save |<ψ(0)|ψ(t)>|
compute_spectrum(cf, dt*stride/fs_to_au)

# Compute reflection and transmission coefficients
if haskey(input, "scatter")
    top_bar = input["top_bar"]
    scatter_T = input["scatter_T"]
    tr_re(data, x_space, top_bar, scatter_T, dt, stride)
end

# Compute electron absorption spectrum
if haskey(input, "vibronic")
    zpe = 1/4/pi*sqrt(k_harm/μ) * fs_to_au*1e15/c    # Compute ZPE
    maxWn = input["vibronic"]["wn_max"]
    compute_spectrum(cf, dt*stride/fs_to_au; maxWn, zpe)
end


# Heller method for resonance Raman excitation profile
# Note: μ_ρ = μ_λ = 1
if haskey(input, "resonanceRaman")
    maxWn = input["resonanceRaman"]["wn_max"]       # Maximal wavenumber in the final spectrum
    zpe = 1/4/pi*sqrt(k_harm/μ) * fs_to_au*1e15/c   # Compute ZPE
    cf = Array{Float64}(undef, N_records)           # Initialize correlation function
    final_state = create_harm_state(1, x_space, x0, k_harm, μ) # Create final scattering states  
    # Compute cross-correlation function:
    for (i, wf_t) in enumerate(eachrow(data))
        cf[i] = abs(dot(final_state, wf_t))
    end
    save_vec(cf, "Raman-CF.dat", dt )
    compute_spectrum(cf, dt*stride/fs_to_au; zpe=zpe, name="resRaman", maxWn=maxWn)
end


# Save eigenstates
println("\nComputing eigenstates:\n")
energies = [150.0, 175.0, 328.0, 366.4 ]
get_eigenfunctions(data, x_space, energies)

# End of program:
println("\nAll done!\n")
