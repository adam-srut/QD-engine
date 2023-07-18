#! /usr/bin/env julia 

using LinearAlgebra, FFTW
using OffsetArrays
using Statistics
using YAML
using NCDatasets
using Printf, Dates, ArgParse
using SpecialPolynomials
using UnicodePlots
using Trapz

# Using more cores for FFTW or BLAS do not enhance performance
FFTW.set_num_threads(1)
BLAS.set_num_threads(1)

#=
    TODO:
    - Object oriented version [x]
        - test 2D dynamics [x]
        - complete NetCDF interface [x]
    - Analyses:
        - CF + IO [x]
        - WF + IO [x]
        - spectra + IO [x]
        - lineshape function [x] 
        - eigenstates + IO (see Feit & Fleck paper)
    - Info section [x]
    - Add benchmark of Nr of grid points [x]
    - Add fail checker - Nyquist + ΔE_min [x]
        - Check for WF at boundaries
    - Add plotting routines [x]
    - Add interpolation scheme [x]
        - Support for irregular grids
    - Special stride for PAmp [x]
    - Read in relaxed ψ(t=0) from NetCDF file [x]
    - Stand alone computation of spectra (various lineshape widths)
       - Frequency shift in Fourier Transform
=#

#=================================================
            Parse arguments
=================================================#

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "input"
            help="YAMl input file"
            arg_type=String
            required=true
    end
    return parse_args(s)
end

args = parse_commandline()
infile = args["input"]

#===================================================
                Basic constants
===================================================#

constants = Dict(
    "wn_to_auFreq" => 7.2516327788e-7,  # cm^-1 to a.u.^-1
    "c_au" => 137.0359991,      	    # Speed of light in a.u.
    "c" => 29979245800,                 # Speed of light in cm/s
    "fs_to_au" => 41.341373335,
    "ang_to_bohr" => 1.889726125,
    "μ_to_me" => 1822.88849,            # Atomic mass unit in masses of an electron
    "e" => 1.60218e-19,                 # e⁻ charge
    "h" => 6.62607e-34,                 # Planck constant
    "kB" => 3.16681e-6,                 # Boltzmann constant in a.u.
    "Eh_to_wn" => 219474.631
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
    wf::NCDataset
    CF::Array{ComplexF64}
    Nsteps::Int
    step_stride::Int
    dt::Number
end

function init_OutData(step_stride::Int, Nsteps::Int, wf0::Array{ComplexF64}, dt::Number)
    #= Initilize output data =#
    N_records = Int(fld(Nsteps, step_stride))
    N_records_WF = Int(fld(Nsteps, step_stride*10)) # Increase stride for saving PAmp
    isfile("WF.nc") && rm("WF.nc")
    if ndims(wf0) == 1
        wfout = NCDataset("WF.nc", "c")
        wfout.attrib["title"] = "File contains temporal evolution of the wavepacket."
        defDim(wfout, "x", length(wf0))
        defDim(wfout, "time", N_records_WF) 
        defVar(wfout, "PAmp", Float32, ("x", "time"), deflatelevel=0)
        defVar(wfout, "Xdim", Float64, ("x",), deflatelevel=0)
        defDim(wfout, "timeCF", N_records)
        defVar(wfout, "CFRe", Float64, ("timeCF", ))
        defVar(wfout, "CFIm", Float64, ("timeCF", ))
    elseif ndims(wf0) == 2
        (nx,ny) = size(wf0)
        wfout = NCDataset("WF.nc", "c")
        wfout.attrib["title"] = "File contains temporal evolution of the wavepacket."
        defDim(wfout, "x", nx)
        defDim(wfout, "y", ny)
        defDim(wfout, "time", N_records_WF) 
        defVar(wfout, "PAmp", Float32, ("x", "y", "time"), deflatelevel=0)
        defVar(wfout, "Xdim", Float64, ("x",), deflatelevel=0)
        defVar(wfout, "Ydim", Float64, ("y",), deflatelevel=0)
        defDim(wfout, "timeCF", N_records)
        defVar(wfout, "CFRe", Float64, ("timeCF", ))
        defVar(wfout, "CFIm", Float64, ("timeCF", ))
    else
        error("Wrong dimensions of the inital WF.")
    end
    cf = Array{ComplexF64}(undef, N_records)
    return OutData(wfout, cf, Nsteps, step_stride, dt)
end

#===================================================
             `MetaData` constructor
===================================================#

struct MetaData
    input::Dict
    potential::Array{Float64}
    ref_potential::Array{Float64}
    xdim::Array{Float64}
    ydim::Array{Float64}
end

function MetaData(;
        inputfile::String="input.yml")
    input = YAML.load_file(inputfile)
    if input["dimensions"] == 1
        (xdim, potential) = read_potential(input["potential"])
        if haskey(input, "imagT")
            (_, ref_potential) = read_potential(input["imagT"]["gs_potential"])
            return MetaData(input, potential, ref_potential, xdim, Array{Float64}(undef, length(xdim)))
        else
            return MetaData(input, potential, Array{Float64}(undef, length(potential)), xdim, Array{Float64}(undef, length(xdim)))
        end
    elseif input["dimensions"] == 2
        (potential, xdim, ydim) = read_potential(input["potential"])
        if haskey(input, "imagT")
            (ref_potential, _, _) = read_potential(input["imagT"]["gs_potential"])
            return MetaData(input, potential, ref_potential, xdim, ydim)
        else
            return MetaData(input, potential, Array{Float64}(undef, length(potential)), xdim, ydim)
        end
    else
        throw(ArgumentError("Dynamics in $(input["dimensions"]) dimensions is not supported."))
    end
end

#===================================================
                Function section
===================================================#

function read_potential(filepath::String)
    #= Reads interpolated potential from a NetCDF file =#
    NCDataset(filepath, "r") do potfile
        if length(potfile.dim) == 1
            xdim = potfile["xdim"][:]
            potential = potfile["potential"][:]
            return (xdim, potential)
        elseif length(potfile.dim) == 2
            xdim = potfile["xdim"][:]
            ydim = potfile["ydim"][:]
            potential = potfile["potential"][:,:]
            return (potential, xdim, ydim)
        else
            throw(DimensionMismatch("Wrong number of dimensions in the $filepath. Expected 1 or 2, got $(length(potfile.dim))."))
        end
    end
end

function read_wf(filepath::String)
    #= Read relaxed wavefunction from NetCDF file =#
    NCDataset(filepath, "r") do wffile
        if !haskey(wffile.dim, "y") 
            wfReal = wffile["wfRe"][:]
            wfImag = wffile["wfIm"][:]
            wf = wfReal .+ 1im*wfImag
        else
            wfReal = wffile["wfRe"][:,:]
            wfImag = wffile["wfIm"][:,:]
            wf = wfReal .+ 1im*wfImag
        end
        return wf
    end
end

function construct_kspace(;xdim, μx, ydim=false, μy=false)
    #= Construct inverse space for applying the kinetic energy operator =#
    Nx = length(xdim)
    Lx = xdim[end] - xdim[1]
    if !(ydim isa Bool)
        Ny = length(ydim)
        Ly = ydim[end] - ydim[1]
        k_space = OffsetArray(Array{Float64}(undef, Nx, Ny), -div(Nx,2)-1, -div(Ny,2)-1)
        for x in -div(Nx,2):div(Nx,2)-1
            for y in -div(Ny,2):div(Ny,2)-1
                k_space[ x, y ] = (1/2/μx)*(2*pi/Lx*x)^2 + (1/2/μy)*(2*pi/Ly*y)^2
            end
        end
    else
        k_space = OffsetArray(Array{Float64}(undef, Nx), -div(Nx,2)-1)
        for x in -div(Nx,2):div(Nx,2)-1
            k_space[ x ] = (1/2/μx) * (2*pi/Lx*x)^2
        end
    end
    return OffsetArrays.no_offset_view(k_space)
end

function propagate!(dynamics::Dynamics)
    #= Propagation with the split operator method
        Returns constructor `dynamics` with modified ψ.
        This is not the "correct" propagator!
        Dynamics has to be initialized with: exp(-i*Δt/2*T̂)ψ(t)
                             and ended with: exp(-i*Δt/2*V̂)ψ(t)  =#
    wf = dynamics.wf
    wf .= wf .* exp.( -(im*dynamics.dt) * dynamics.potential )
    wf .= dynamics.PFFT * wf
    wf .= fftshift(wf)
    wf .= wf .* exp.( -(im*dynamics.dt) * dynamics.k_space )
    wf .= fftshift(wf)
    wf .= dynamics.PIFFT * wf
    dynamics.wf .= wf
    wf = nothing
    return dynamics
end

function T_halfstep!(dynamics::Dynamics)
    #= Initialize dynamics with a half-step propagation 
       of a free particle: exp(-i*Δt/2*T̂)ψ(t) =#
    wf = dynamics.wf
    wf .= dynamics.PFFT * wf
    wf .= fftshift(wf)
    wf .= wf .* exp.( -(im*dynamics.dt/2) * dynamics.k_space )
    wf .= fftshift(wf)
    wf .= dynamics.PIFFT * wf
    dynamics.wf .= wf
    wf = nothing
    return dynamics
end

function V_halfstep!(dynamics::Dynamics)
    #= End dynamics with a half-step phase change: 
        exp(-i*Δt/2*V̂)ψ(t) =#
    wf = dynamics.wf
    wf .= wf .* exp.( -(im*dynamics.dt/2) * dynamics.potential )
    dynamics.wf .= wf
    wf = nothing
    return dynamics
end

function create_harm_state(n::Int, x_space::Array{Float64}, x0::Number,
        ω::Number, μ::Number)
    #= Function returns a harmonic vibrational level. =#
    k = (ω*2*pi)^2 * μ
    chi(n::Int, x::Float64) = 1/(sqrt(2^n*factorial(n))) * 
        basis(Hermite, n)((μ*k)^(1/4)*(x-x0)) * 
        exp( -(1/2*sqrt(μ*k) * (x-x0)^2) )
    wf = [ chi(n, i) for i in x_space ]
    wf = wf / sqrt(dot(wf, wf)) .+ 0*im
    return wf
end

function create_harm_state_2D(;n::Int, xdim::Array{Float64}, ydim::Array{Float64}, 
        x0::Number, y0::Number, ωx::Number, μx::Number, ωy::Number, μy::Number)
    #= Function returns a harmonic vibrational level. =#
    (kx, ky) = ((ωx*2*pi)^2 * μx, (ωy*2*pi)^2 * μy)
    chix(n::Int, x::Float64) = 1/(sqrt(2^n*factorial(n))) *
        basis(Hermite, n)((μx*kx)^(1/4)*(x-x0)) *
        exp( -(1/2*sqrt(μx*kx) * (x-x0)^2) )
    chiy(n::Int, y::Float64) = 1/(sqrt(2^n*factorial(n))) *
        basis(Hermite, n)((μy*ky)^(1/4)*(y-y0)) *
        exp( -(1/2*sqrt(μy*ky) * (y-y0)^2) )
    wf = vcat([ [chix(n, x)*chiy(n, y) for x in xdim] for y in ydim ]'...)
    wf = wf .+ 0*im
    wf = wf / sqrt(dot(wf,wf))
    return wf
end


function imTime_propagation(dynamics::Dynamics; Nsteps::Int=5000)
    #= Function will propagate an arbitrary WF in imaginary time.
        Use to determine the initial WP. 
        WF is renormalized in the middle of the integration step. 
         =#

    # Set up progress bar
    print("\t  Progress:\n\t   ")
    
    dynamics.dt = -dynamics.dt*im
    T_halfstep!(dynamics)

    while dynamics.istep < Nsteps
        
        propagate!(dynamics)
        WFnorm = sqrt(dot(dynamics.wf, dynamics.wf))
        dynamics.wf .= dynamics.wf / WFnorm
        dynamics.istep += 1

        # Update progress bar
        if dynamics.istep in round.(Int, range(start=1, stop=Nsteps, length=10))
            print( @sprintf "%2d%%" 100*dynamics.istep/Nsteps)
            flush(stdout)
        end
    end
    print("\n")

    V_halfstep!(dynamics)
    dynamics.dt = Float64(dynamics.dt*im)
    dynamics.istep = 0
    
    (energy, Epot, Ekin) = compute_energy(dynamics, metadata)
    println(@sprintf "\n\t  Energy of the WP:%10.2f cm^-1." energy)

    # Save initial condition:
    isfile("initWF.nc") && rm("initWF.nc")
    NCDataset("initWF.nc", "c") do outfile
        outfile.attrib["title"] = "File contains inital WF from imaginary time propagation."
        outfile.attrib["energy"] = "energy of the relaxed WF is: $energy cm⁻¹."
        if metadata.input["dimensions"] == 1
            defDim(outfile, "x", length(dynamics.wf))
            defDim(outfile, "scalars", 1)
            defVar(outfile, "wfRe", Float64, ("x",))
            defVar(outfile, "wfIm", Float64, ("x",))
            defVar(outfile, "xdim", Float64, ("x",))
            defVar(outfile, "ZPE", Float64, ("scalars",))
            defVar(outfile, "ZPE_T", Float64, ("scalars",))
            defVar(outfile, "ZPE_V", Float64, ("scalars",))
            outfile["ZPE"][1] = energy
            outfile["ZPE_T"][1] = Ekin
            outfile["ZPE_V"][1] = Epot
            outfile["wfRe"][:] = real.(dynamics.wf)
            outfile["wfIm"][:] = imag.(dynamics.wf)
            outfile["xdim"][:] = metadata.xdim
        elseif metadata.input["dimensions"] == 2
            outfile.attrib["title"] = "File contains inital WF from imaginary time propagation."
            (nx, ny) = size(dynamics.wf)
            defDim(outfile, "x", nx)
            defDim(outfile, "y", ny)
            defDim(outfile, "scalars", 1)
            defVar(outfile, "xdim", Float64, ("x",))
            defVar(outfile, "ydim", Float64, ("y",))
            defVar(outfile, "wfRe", Float64, ("x", "y"))
            defVar(outfile, "wfIm", Float64, ("x", "y"))
            defVar(outfile, "ZPE", Float64, ("scalars",))
            defVar(outfile, "ZPE_T", Float64, ("scalars",))
            defVar(outfile, "ZPE_V", Float64, ("scalars",))
            outfile["ZPE"][1] = energy
            outfile["ZPE_T"][1] = Ekin
            outfile["ZPE_V"][1] = Epot
            outfile["wfRe"][:,:] = real.(dynamics.wf)
            outfile["wfIm"][:,:] = imag.(dynamics.wf)
            outfile["xdim"][:] = metadata.xdim
            outfile["ydim"][:] = metadata.ydim
        end
    end
    println("\t  Relaxed wavefunction saved to: `initWF.nc`")
    flush(stdout)
    return dynamics
end

function execute_dynamics(dynamics::Dynamics, outdata::OutData)
    #= Execute dynamics with the predefined setup.
        Returns object `outdata` with ψ(tᵢ) and <ψ(0)|ψ(tᵢ)>
        tᵢ = Δt * step_stride * istep =#

    # Set up progress bar
    print("\t  Progress:\n\t    0%")
    flush(stdout) 

    # t=0
    wf0 = copy(dynamics.wf)
    T_halfstep!(dynamics) # Initialize
    dynamics.istep += 1 # t0 + Δt/2
    
    # Propagation:
    while dynamics.istep < dynamics.Nsteps

        # Propagate
        propagate!(dynamics)
        dynamics.istep += 1 # t + 3/2*Δt
 
        # Update progress bar
        if dynamics.istep in round.(Int, range(start=1, stop=dynamics.Nsteps, length=10))
            print( @sprintf "%2d%%" 100*dynamics.istep/dynamics.Nsteps )
            flush(stdout)
        end
       
        # Compute CF and save PAmp:
        if mod(dynamics.istep, dynamics.step_stride) == 0
            V_halfstep!(dynamics) # Complete integration
            irec = div(dynamics.istep, dynamics.step_stride)
            overlap = dot(wf0, dynamics.wf)
            outdata.CF[irec] = overlap
            outdata.wf["CFRe"][irec] = real(overlap)
            outdata.wf["CFIm"][irec] = imag(overlap)
           
            # Save probability amplitude with 10*stride
            if mod(dynamics.istep, 10*dynamics.step_stride) == 0
                irec = div(dynamics.istep, 10*dynamics.step_stride)
                if metadata.input["dimensions"] == 1
                    outdata.wf["PAmp"][:, irec] = Float32.(conj.(dynamics.wf) .* dynamics.wf)
                elseif metadata.input["dimensions"] == 2
                    outdata.wf["PAmp"][:, :, irec] = Float32.(conj.(dynamics.wf) .* dynamics.wf)
                end
                #GC.gc() # call garbage collector
            end

            T_halfstep!(dynamics) # Initialize new step
            dynamics.istep += 1
        end
    end
    V_halfstep!(dynamics) # Complete integration
    return outdata
end

#===================================================
                Prepare dynamics 
===================================================#

starttime = now()
println("\n\tStarted " * Dates.format(starttime, "on dd/mm/yyyy at HH:MM:SS") * "\n")
flush(stdout)

# Include helper functions:
include("analysis.jl")
include("info.jl")
# Echo hello
print_hello()

# Essential parameters from "input.yml":
metadata = MetaData(inputfile=infile)

if metadata.input["dimensions"] == 1
    # Read-in data and settings:
    # Prepare ψ(t=0):
    if haskey(metadata.input["initWF"], "fromfile")
        wf0 = read_wf(metadata.input["initWF"]["fromfile"])
    else
        wf0 = create_harm_state(0, metadata.xdim, 
                                metadata.input["initWF"]["initpos"], 
                                metadata.input["initWF"]["freq"]*constants["wn_to_auFreq"], 
                                metadata.input["mass"]*constants["μ_to_me"])   
    end
    # Prepare dynamics:
    k_space = construct_kspace(xdim=metadata.xdim, 
                               μx=metadata.input["mass"]*constants["μ_to_me"])
    dynamics = Dynamics(potential=metadata.potential, 
                        k_space=k_space, 
                        wf=wf0,
                        dt=metadata.input["params"]["dt"], 
                        step_stride=metadata.input["params"]["stride"], 
                        Nsteps=metadata.input["params"]["Nsteps"])
    outdata = init_OutData(metadata.input["params"]["stride"], 
                           metadata.input["params"]["Nsteps"], 
                           wf0, 
                           metadata.input["params"]["dt"])
    outdata.wf["Xdim"][:] = metadata.xdim
elseif metadata.input["dimensions"] == 2
    # Prepare ψ(t=0):
    if haskey(metadata.input["initWF"], "fromfile")
        wf0 = read_wf(metadata.input["initWF"]["fromfile"])
    else
        wf0 = create_harm_state_2D(n=0, xdim=metadata.xdim, ydim=metadata.ydim,
                               x0=metadata.input["initWF"]["initpos"][1],
                               y0=metadata.input["initWF"]["initpos"][2],
                               ωx=metadata.input["initWF"]["freq"][1]*constants["wn_to_auFreq"],
                               μx=metadata.input["mass"][1]*constants["μ_to_me"],
                               ωy=metadata.input["initWF"]["freq"][2]*constants["wn_to_auFreq"],
                               μy=metadata.input["mass"][2]*constants["μ_to_me"])
    end
    # Prepare dynamics:
    k_space = construct_kspace(xdim=metadata.xdim, 
                               μx=metadata.input["mass"][1]*constants["μ_to_me"],
                               ydim=metadata.ydim,
                               μy=metadata.input["mass"][2]*constants["μ_to_me"])
    dynamics = Dynamics(potential=metadata.potential,
                        k_space=k_space,
                        wf=wf0,
                        dt=metadata.input["params"]["dt"],
                        step_stride=metadata.input["params"]["stride"],
                        Nsteps=metadata.input["params"]["Nsteps"])
    outdata = init_OutData(metadata.input["params"]["stride"],
                           metadata.input["params"]["Nsteps"],
                           wf0,
                           metadata.input["params"]["dt"])
    outdata.wf["Xdim"][:] = metadata.xdim
    outdata.wf["Ydim"][:] = metadata.ydim
end

# Echo init
print_init(metadata)

#===================================================
            Imaginary time propagation
===================================================#

if haskey(metadata.input, "imagT")
    println("\t========> Imaginary time propagation <=========")
    #@info "Assuming the same grid for reference and actual potential.\n"
    dynamics.potential = metadata.ref_potential
    dynamics = imTime_propagation(dynamics)
    dynamics.potential = metadata.potential
end


#===================================================
                Execute dynamics
===================================================#

print_run()

outdata = execute_dynamics(dynamics, outdata)

println("\n")

#===================================================
            Analyze and save the results
===================================================#

compute_spectrum(outdata)
save_CF(outdata)

# Save gnuplot scripts
GP_spectrum()
GP_correlation_function()

# Close NetCDF file with PAmp:
close(outdata.wf)

print_output()

# End of the program:
endtime = now()
println("\n  Finished successfully " * Dates.format(endtime, "on dd/mm/yyyy at HH:MM:SS") )
println("  Total runtime: " * Dates.format(convert(DateTime, endtime-starttime ), "HH:MM:SS") * "\n")
