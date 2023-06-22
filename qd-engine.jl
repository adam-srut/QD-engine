#! /usr/bin/env -S julia --project=/work/qd-engine/project_1dqd

println("Loading packages...")
using LinearAlgebra, FFTW
using OffsetArrays
using Statistics
using YAML
using Printf
using SpecialPolynomials
using UnicodePlots
include("analysis.jl")
include("info.jl")

FFTW.set_num_threads(1)

#=
  TODO:
    - Object oriented version
    - Try relaxation method to find eigenstates.
    - Use proper normalization, e.g. trapezoidal rule (not necessary)
    - Add benchmark of Nr of grid points
    - Add fail checker - Nyquist + ΔE_min
=#

#===================================================
                Basic constants
===================================================#

constants = Dict(
    "wn_to_au" => 5.29177210903e-9,
    "c_au" => 137.0359991,      	    # Speed of light in a.u.
    "c" => 29979245800,                 # Speed of light in cm/s
    "fs_to_au" => 41.341373335,
    "ang_to_bohr" => 1.889726125,
    "μ_to_me" => 1822.88849,            # Atomic mass unit in masses of an electron
    "e" => 1.60218e-19,                 # e⁻ charge
    "h" => 6.62607e-34,                 # Planck constant
    "kB" => 3.16681e-6                  # Boltzmann constant in a.u.
   )

#===================================================
             `Dynamics` constructor
===================================================#

mutable struct Dynamics
    potential::Array{Float64}   # Electronic potential
    k_space::Array{Float64}     # momentum space: (1/2/μ) * (2*π/L * k_i)^2
    wf::Array{ComplexF64}       # wavefunction 
    dt::Number                  # integration step in a.u.
    PFFT                        # planned forward FT
    PIFFT                       # planned inverse FT
    step_stride::Int            # stride for saving the results
    Nsteps::Int                 # Maximum number of steps
    istep::Int                  # Current step
end

function Dynamics(;
        #= Setting up the default values. =#
        potential = Array{Float64},
        k_space = Array{Float64},
        wf = Array{ComplexF64},
        dt = 1,
        PFFT = plan_fft(wf),
        PIFFT = plan_ifft(wf),
        step_stride = 10,
        Nsteps = 10000,
        istep = 0
    )
    return Dynamics(potential, k_space, wf, dt, PFFT, PIFFT, step_stride, Nsteps, istep)
end

#===================================================
             `OutData` constructor
===================================================#

mutable struct OutData
    wf
    CF
end

function init_OutData(step_stride::Int, Nsteps::Int, wf0::Array{ComplexF64})
    #= Initilize output data =#
    N_records = Int(fld(Nsteps, step_stride)) + 1
    if ndims(wf0) == 1
        wfout =  OffsetArray(Array{ComplexF64}(undef, N_records, length(wf0)), -1, 0)
    elseif ndims == 2
        (nx,ny) = size(wf0)
        wfout = OffsetArray(Array{ComplexF64}(undef, N_records, nx, ny), -1, 0, 0)
    else
        error("Wrong dimensions of the inital WF.")
    end
    cf = OffsetArray(Array{Float64}(undef, N_records), -1)
    return OutData(wfout, cf)
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
            if startswith(line, "#")
                continue
            end
            line = parse.(Float64, split(line))
            append!(x_axis, line[1])
            append!(potential, line[2])
        end
        return (x_axis, potential)
    end
end

function read_potential2D(filepath::String, state::Int=3)
	#= Read 2D potential in format:
	 x_pos | y_pos | ... | state index | ... =#
	pot = []
	open(filepath) do file
		for line in eachline(file)
			if startswith(line, "#")
				continue
			end
			line = split(line)
			vals = map( x -> parse(Float64, line[x]), [1,2, state])
			append!(pot, [vals])
		end
	end
	x_dim = sort(unique(map( x -> x[1], pot)))
	N_x = length(x_dim)
    dx = x_dim[2]-x_dim[1]
	y_dim = sort(unique(map( x -> x[2], pot)))
	N_y = length(y_dim)
    dy = y_dim[2]-y_dim[1]
    potential = Array{Float64}(undef, N_x, N_y)
    for line in pot
        x = line[1]
        x_i = round(Int, (x - x_dim[1])/dx) + 1
        y = line[2]
        y_i = round(Int, (y - y_dim[1])/dy) + 1
        potential[x_i, y_i] = line[3]
    end
    return (potential, x_dim, y_dim)
end

function construct_kspace(;x_dim, μx, y_dim=false, μy=false)
    #= Construct inverse space for applying the kinetic energy operator =#
    N_x = length(x_dim)
    L_x = x_dim[end] - x_dim[1]
    if !(y_dim isa Bool)
        N_y = length(y_dim)
        L_y = y_dim[end] - y_dim[1]
        k_space = Array{Float64}(undef, N_x, N_y)
        for x in -div(N_x,2):div(N_x,2)-1
            for y in -div(N_y,2):div(N_y,2)-1
                k_space[x + div(N_x,2) + 1, y + div(N_x,2) + 1] = (1/2/μx)*(2*pi/L_x*x)^2 + (1/2/μy)*(2*pi/L_y*y)^2
            end
        end
    else
        k_space = Array{Float64}(undef, N_x)
        for x in -div(N_x,2):div(N_x,2)-1
            k_space[x + div(N_x,2) + 1] = (1/2/μx) * (2*pi/L_x*x)^2
        end
    end
    return k_space
end

function propagate(dynamics::Dynamics)
    #= Propagation with the split operator method
        Returns constructor `dynamics` with modified ψ.
        This is not the "correct" propagator!
        Dynamics has to be initialized with: exp(-i*Δt/2*T̂)ψ(t)
                             and ended with: exp(-i*Δt/2*V̂)ψ(t)  =#
    wf = dynamics.wf
    wf = wf .* exp.( -(im*dynamics.dt) * dynamics.potential )
    wf = dynamics.PFFT * wf
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*dynamics.dt) * dynamics.k_space )
    wf = fftshift(wf)
    wf = dynamics.PIFFT * wf
    dynamics.wf = wf
    return dynamics
end

function T_halfstep(dynamics::Dynamics)
    #= Initialize dynamics with half-step propagation 
       of a free particle: exp(-i*Δt/2*T̂)ψ(t) =#
    wf = dynamics.wf
    wf = dynamics.PFFT * wf
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*dynamics.dt/2) * dynamics.k_space )
    wf = fftshift(wf)
    wf = dynamics.PIFFT * wf
    dynamics.wf = wf
    return dynamics
end

function V_halfstep(dynamics::Dynamics)
    #= End dynamics with half-step propagation 
       of a free particle: exp(-i*Δt/2*V̂)ψ(t) =#
    wf = dynamics.wf
    wf = wf .* exp.( -(im*dynamics.dt/2) * dynamics.potential )
    dynamics.wf = wf
    return dynamics
end

function create_harm_state(n::Int, x_space::Array{Float64}, x0::Number,
        k::Number, μ::Number)
    #= Function returns the harmonic vibrational level.
     Force constant k, mass μ and centre x0 have to be specified. =#
    chi(n::Int, x::Float64) = 1/(sqrt(2^n*factorial(n))) * 
        basis(Hermite, n)((μ*k)^(1/4)*(x-x0)) * 
        exp( -(1/2*sqrt(μ*k) * (x-x0)^2) )
    wf = [ chi(n, i) for i in x_space ]
    wf = wf / sqrt(dot(wf, wf)) .+ 0*im
    return wf
end

function imTime_propagation(dynamics::Dynamics, Nsteps::Int=5000)
    #= Function will propagate an arbitrary WF in imaginary time.
        Use to determine the initial WP. =#

    # Set up progress bar
    print("\t  Progress:\n\t    0%")
    
    dynamics.dt = -dynamics.dt*im
    dynamics = T_halfstep(dynamics)

    while dynamics.istep < Nsteps
        
        dynamics = propagate(dynamics)
        WFnorm = sqrt(dot(dynamics.wf, dynamics.wf))
        dynamics.wf = dynamics.wf / WFnorm
        dynamics.istep += 1

        # Update progress bar
        if dynamics.istep in round.(Int, range(start=1, stop=Nsteps, length=10))
            print( @sprintf "%2d%%" 100*dynamics.istep/Nsteps)
        end
    end

    dynamics = V_halfstep(dynamics)
    dynamics.dt = dynamics.dt*im

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

function execute_dynamics(dynamics::Dynamics, outdata::OutData)
    #= Execute dynamics with predefined setup.
        Object `outdata` is return containing ψ(tᵢ) and <ψ(0)|ψ(tᵢ)>
        tᵢ = Δt * step_stride * istep =#

    # Set up progress bar
    print("\t  Progress:\n\t    0%")
    
    # t=0
    input["dimensions"] == 1 ? outdata.wf[0,:] = dynamics.wf : outdata.wf[0,:,:] = dynamics.wf
    outdata.CF[0] = 1
    wf0 = dynamics.wf
    dynamics = T_halfstep(dynamics)
    dynamics.istep += 1 # t0 + Δt/2
    
    # Propagation:
    while dynamics.istep < dynamics.Nsteps

        # Propagate
        dynamics = propagate(dynamics)
        dynamics.istep += 1 # t + 3/2*Δt
 
        # Update progress bar
        if dynamics.istep in round.(Int, range(start=1, stop=dynamics.Nsteps, length=10))
            print( @sprintf "%2d%%" 100*dynamics.istep/dynamics.Nsteps)
        end
       
        # Save data:
        if mod(dynamics.istep, step_stride) == 0
            dynamics = V_halfstep(dynamics)
            nrec = div(dynamics.istep, dynamics.step_stride)
            if input["dimensions"] == 1
                outdata.wf[nrec, :] = dynamics.wf
            else
                outdata.wf[nrec, :, :] = dynamics.wf
            end
            outdata.CF[nrec] = abs(dot(wf0, dynamics.wf))

            dynamics = T_halfstep(dynamics)
            dynamics.istep += 1
        end
    end
    return outdata
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
            t = (time-1)*step_stride*dt
            eigenstate = eigenstate + wp*exp(-im*freq*t)
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
                line *= @sprintf "%14.4f%14.4f" real(val) abs(val)
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
if input["dimensions"] == 1
    # Read-in data and settings:
    (x_dim, potential) = read_potential(input["potential"])
    μ = input["mass"] * constants["μ_to_me"]            # Mass in a.u.
    dt = input["params"]["dt"]
    step_stride = input["params"]["stride"]
    Nsteps = input["params"]["Nsteps"]
    # Prepare ψ(t=0):
    wf0 = create_harm_state(0, x_dim, input["initWF"]["initpos"], input["initWF"]["k_harm"], μ)   
    # Prepare dynamics:
    k_space = construct_kspace(x_dim=x_dim, μx=μ)
    dynamics = Dynamics(potential=potential, k_space=k_space, wf=wf0, dt=dt, step_stride=step_stride, Nsteps=Nsteps)
    outdata = init_OutData(step_stride, Nsteps, wf0)
end


# Echo input
#print_init()
#===================================================
            Imaginary time propagation
===================================================#

if haskey(input, "imagT")
    println("\tImaginary time propagation...")
    (x_space, gs_potential) = read_potential(input["imagT"]["gs_potential"])
    if haskey(input["imagT"], "dt") && haskey(input["imagT"], "tmax")
        dtim = input["imagT"]["dt"]
        tmaxim = input["imagT"]["tmax"]
        wf0 = imTime_propagation(wf0, gs_potential, dtim, tmaxim)
        e = abs(compute_energy(wf0, gs_potential, p))/fs_to_au*1e15/c
        println(e)
    else
        wf0 = imTime_propagation(wf0, gs_potential)
#        println(compute_energy(wf0, gs_potential, p))
    end
end


#===================================================
                Execute dynamics
===================================================#

#print_run()

outdata = execute_dynamics(dynamics, outdata)

println("\n")
error("You shall not pass.")

#===================================================
            Analyze and save the results
===================================================#

print_analyses()
save_data(data, x_space, dt)            # Save |Ψ(x,t)|² and |Ψ(k,t)|²
save_vec(cf, "corrF.dat", dt )          # Save |<ψ(0)|ψ(t)>|
compute_spectrum(cf, dt*step_stride/fs_to_au)

## Compute electron absorption spectrum
#if haskey(input, "vibronic")
#    zpe = 1/4/pi*sqrt(k_harm/μ) * fs_to_au*1e15/c    # Compute ZPE
#    maxWn = input["vibronic"]["wn_max"]
#    compute_spectrum(cf, dt*step_stride/fs_to_au; maxWn, zpe)
#end



## Save eigenstates
#println("\nComputing eigenstates:\n")
#energies = [0.0, 50, 103.0, 310.0, 626.0, 2082.0 ]
#get_eigenfunctions(data, x_space, energies)

# End of program:
println("\nAll done!\n")
