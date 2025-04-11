#!/usr/bin/env julia

#==================================================
**************************************************
****    Fourier Grid Hamiltonian Tool         ****
**************************************************
**** Created at TU Darmstadt in Krewald Group.****
**************************************************
**** Author: Adam Šrut                        ****
**************************************************
**************************************************
**** The script is a standalone part of       ****
****   QD-Engine package.                     ****
****   Reads `input.yml`                      ****
****   Constructs Hamiltoninan in DVR,        ****
****   returns eigenenergies and eigenstates  ****
****   in `eigenstates.nc` file.              ****
**************************************************
==================================================#

#==================================================
            Example YAML input file:

# Number of dimensions:
dimensions : 1

# File with a potential (expected in NetCDF format)
potential : "GS_pot.nc"

# Mass of the particle (in amu)
mass : 14.006  # for 2D use [14.006, 14.006]

# Fouried Grid Hamiltonian method:
FGH:
    Nmax: 5               # Number of eigenvalues and eigenstates to calculate
    method: "iterative"   # Method for matrix diagonalization (see below)
    # advanced:                  # Advanced options 
    #     precision: "Double"    # Precision of the matrix elements ("Double" or "Single")
    #     krylovdim: 100         # Dimension of the Krylov subspace
    #     tol: 12                # Requested accuracy
    #     maxiter: 100           # Maximum number of iterations
    #     verbosity: 1           # Verbosity level (1-only warnings; 3-info after every iteration) 

# User can choose between "Iterative" and "Exact" method for diagonalization of Hamiltonian
#   Iterative refers to Lanzcos algorithm using KrylovKit package.
#   Exact diagonalization is suitable for small grids (i.e. 1D potentials)

===================================================#

# Load necessary packages:
using LinearAlgebra, FFTW, KrylovKit
using OffsetArrays
using Statistics
using YAML
using NCDatasets
using Printf, Dates, ArgParse
using Trapz
using Base.Threads

# Set number of threads for numerical routines:
# Keep FFTW serial; parallelize outer loop for H construction: `construct_T` function
FFTW.set_num_threads(1)
BLAS.set_num_threads(Threads.nthreads()) 
KrylovKit.set_num_threads(Threads.nthreads())

#=================================================
            Parse arguments
=================================================#

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "input"
            help="YAML input file"
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
    "μ_to_me" => 1822.88849,            # Atomic mass unit to atomic units
    "e" => 1.60218e-19,                 # e⁻ charge
    "h" => 6.62607e-34,                 # Planck constant
    "kB" => 3.16681e-6,                 # Boltzmann constant in a.u.
    "Eh_to_wn" => 219474.631
    )

#===================================================
             `MetaData` constructor
===================================================#
# Collect input file, potential, k-space and spation dimensions
#   into a single object

struct MetaData
    input::Dict
    potential::Array{Float64}
    k_space::Array{Float64}
    xdim::Array{Float64}
    ydim::Array{Float64}
end

function MetaData(; inputfile::String="input.yml")
    input = YAML.load_file(inputfile)
    if input["dimensions"] == 1
        (xdim, potential) = read_potential(input["potential"])
        k_space = construct_kspace(xdim=xdim, μx=input["mass"]*constants["μ_to_me"])
        return MetaData(input, potential, k_space, xdim, Array{Float64}(undef, length(xdim)))
    elseif input["dimensions"] == 2
        (potential, xdim, ydim) = read_potential(input["potential"])
        # if kinetic coupling is not provided, set it to zero:
        haskey(input, "kcoup") ? kcoup = input["kcoup"] : kcoup = 0.0
        k_space = construct_kspace(xdim=xdim, μx=input["mass"][1]*constants["μ_to_me"], ydim=ydim, μy=input["mass"][2]*constants["μ_to_me"], kcoup=kcoup)
        return MetaData(input, potential, k_space, xdim, ydim)
    else
        throw(ArgumentError("Dynamics in $(input["dimensions"]) dimensions is not supported."))
    end
end

#===================================================
                Function section
===================================================#

function construct_kspace(;xdim, μx, ydim=false, μy=false, kcoup::Float64=0.0)
    #= Construct inverse space for applying the kinetic energy operator =#
    Nx = length(xdim)
    Lx = xdim[end] - xdim[1]
    # 2D dynamics:
    if !(ydim isa Bool)
        Ny = length(ydim)
        Ly = ydim[end] - ydim[1]
        k_space = OffsetArray(Array{Float64}(undef, Nx, Ny), -div(Nx,2)-1, -div(Ny,2)-1)
        for x in -div(Nx,2):div(Nx,2)-1
            for y in -div(Ny,2):div(Ny,2)-1
                # Diagonal term of the G-tensor
                k_space[ x, y ] = (1/2/μx)*(2*pi/Lx*x)^2 + (1/2/μy)*(2*pi/Ly*y)^2
                # mixed term:
                k_space[ x, y ] += 1/2*kcoup * (2*pi)^2 * (1/Ly/Lx) * x*y
            end
        end
    # 1D dynamics:
    else
        k_space = OffsetArray(Array{Float64}(undef, Nx), -div(Nx,2)-1)
        for x in -div(Nx,2):div(Nx,2)-1
            k_space[ x ] = (1/2/μx) * (2*pi/Lx*x)^2
        end
    end
    k_space = OffsetArrays.no_offset_view(k_space)
    # Genereate the offset with fftshift:
    #   now k_space can be directly multiplied with FT[|x>]
    k_space = fftshift(k_space)
    return k_space
end

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

function construct_ith_column(k_space::Union{Array, Matrix}, i::Int, pfft, pifft)
    #= Construct the i-th column of the kinetic energy operator 
        Requires planned FFT and IFFT. =#
    # Prepare one-hot vector:
    phi = zeros(size(k_space)...)
    phi[i] = 1
    # Calculate i-th column of the T operator:
    Tx = pifft * (k_space .* (pfft * phi))
    # Unwrap the results into a vector for 2D:
    if ndims(Tx) == 2 
        return vcat(Tx...)
    elseif ndims(Tx) == 1
        return Tx
    else
        error("Wrong number of dimensions. Expects 1 or 2, got: $(ndims(Tx))")
    end
end

function construct_T(k_space::Union{Array, Matrix}, metadata::MetaData)
    #= Construct the kinetic energy operator =#
    N = length(k_space)
    # Initialize the kinetic energy operator 
    if lowercase.(metadata.input["FGH"]["advanced"]["precision"]) == "single"
        T = zeros(ComplexF32, N,N)
        k_space = Float32.(k_space) 
    elseif lowercase.(metadata.input["FGH"]["advanced"]["precision"]) == "double"
        T = zeros(ComplexF64, N,N) 
    else
        throw(ArgumentError("Unknown precision. Use 'Double' or 'Single'."))
    end
    # Plan forward and inverse FFT:
    pfft = plan_fft(k_space)
    pifft = plan_ifft(k_space)
    # Construct the kinetic energy operator column by column:
    #   Use emabarassingly parallel approach with @threads
    @threads for icol in 1:N
        Tcol = construct_ith_column(k_space, icol, pfft, pifft)
        T[:,icol] .= Tcol
    end
    return T
end

function construct_H(metadata::MetaData)
    #= Construct the Hamiltonian operator =#
    k_space = metadata.k_space
    pot = metadata.potential
    # Convert potential to single-precision if requested:
    if lowercase.(metadata.input["FGH"]["advanced"]["precision"]) == "single"
        pot = Float32.(pot)
        k_space = Float32.(k_space)
    end
    # Kinetic energy operator in matrix representation:
    T = construct_T(k_space, metadata)
    # Potential energy operator in matrix representation:
    V = Diagonal(vcat(pot...))
    H = T .+ V
    # return H as Hermitian matrix:
    H = Hermitian(H)
    return H
end

function exact_diagonalization(H::Union{Hermitian, Matrix}, nmax::Int)
    #= Exact diagonalization of the Hamiltonian operator 
        Calculates all eigenvalues and eigenvectors, but returns only `nmax`. =#
    H = Hermitian(H)
    result = @timed eigen(H)
    F = result.value
    # Print runtime info:
    println("\n\t==========> Diagonalization finished! <==========")
    println("""\tExact diagonalization run overview:
        \t  Time needed $(round(result.time)) seconds
        \t  Memory used $(round(result.bytes / 1024^3, digits=2)) GB
    """)
    return (F.values[1:nmax], F.vectors[:,1:nmax])
end

function Lanczos_algorithm(H::Union{Hermitian, Matrix}, nmax::Int;
        tol::Float64=KrylovDefaults.tol, 
        maxiter::Int=KrylovDefaults.maxiter, 
        krylovdim::Int=100,
        verbosity::Int=1) 
    #= Iterative Lanczos algorithm for diagonalization of the Hamiltonian operator
        using KrylovKit package =#
    result = @timed eigsolve(H,
        nmax,                   # calculate nmax eigenvalues
        :SR,                    # search for smallest eigenvalues
        tol=tol,                # set tolerance (default: 1e-12)
        maxiter=maxiter,        # set maximum number of iterations (default: 100)
        krylovdim=krylovdim,    # set the dimension of the Krylov subspace (default: 100)
        ishermitian=true,
        verbosity=verbosity)
    (vals, vecs, info) = result.value
    # Collect vectors into a Matrix (sorted as columns):
    vecs = Matrix(vcat(vecs[1:nmax]'...)')
    vals = vals[1:nmax]
    # Print runtime info:
    println("\n\t==========> Diagonalization finished! <==========")
    println("\tLanczos algorithm run overview:")
    println(@sprintf "\t  %-32s%10d" "Number of iterations" info.numiter)
    println(@sprintf "\t  %-32s%10d" "Number of converged eigenvalues" info.converged)
    println(@sprintf "\t  %-32s%10d" "Number of Krylov vectors" krylovdim)
    println(@sprintf "\t  %-32s%10d s" "Time needed" result.time) 
    println(@sprintf "\t  %-32s%10.2f GB" "Total memory allocated" result.bytes / 1024^3) 
    # return the eigenvalues and eigenvectors:
    return (vals, vecs)
end

function renormalize_vectors(vecs::Matrix, metadata::MetaData)
    #= Renormalize the eigenvectors =#
    if metadata.input["dimensions"] == 1
        (Nx, NVecs) = size(vecs)
        vecs_rnm = copy(vecs)
        for ivec in 1:NVecs
            state = vecs[:, ivec]
            # calculate the norm:
            n = trapz(metadata.xdim, conj.(state) .* state )
            vecs_rnm[:, ivec] = state/n
        end
        return vecs_rnm
    elseif metadata.input["dimensions"] == 2
        (Nx, Ny) = size(metadata.k_space)
        (_, NVecs) = size(vecs)
        vecs_rnm = Array{ComplexF64}(undef, Nx, Ny, NVecs)
        for ivec in 1:NVecs
            state = reshape(vecs[:, ivec], (Nx, Ny))
            # calculate the norm:
            n = trapz((metadata.xdim, metadata.ydim), conj.(state) .* state )
            vecs_rnm[:, :, ivec] = state/n
        end
        return vecs_rnm
    end
end

function prepare_inp_param(metadata::MetaData)
    #= Prepare input parameters for the FGH algorithm =#
    advancedParams = Dict(
            "precision" => "Double",  # Precision of the calculation
            "tol" => 12,              # Tolerance for iterative methods
            "maxiter" => 100,         # Maximum number of iterations
            "krylovdim" => 100,       # Krylov subspace dimension
            "verbosity" => 1          # Verbosity level
        )
    # Check if the user provided advanced parameters:
    if haskey(metadata.input["FGH"], "advanced")
        # Merge default parameters with user-defined parameters
        for (key, value) in metadata.input["FGH"]["advanced"]
            advancedParams[key] = value
        end
    end
    # Merge advanced parameters with default parameters:
    metadata.input["FGH"]["advanced"] = advancedParams
    return metadata
end

function execute_FGH(metadata::MetaData)
    #= Execute the FGH algorithm =#
    # Prepare Hamiltonian:
    println("\tConstructing Hamiltonian...")
    flush(stdout)
    resH = @timed construct_H(metadata)
    H = resH.value
    # Print Hamiltonian info:
    println(@sprintf "\t  %-28s%8d × %d" "Hamiltonian matrix size:" size(H,1) size(H,2))
    println(@sprintf "\t  %-28s%8d s" "Time for construction:" resH.time)
    if lowercase.(metadata.input["FGH"]["advanced"]["precision"]) == "single"
        println(@sprintf "\t  %-28s%8.2f GB" "Size in memory:" (size(H,1)^2*8 / 1024^3))
    else
        println(@sprintf "\t  %-28s%8.2f GB" "Size in memory:" (size(H,1)^2*16 / 1024^3))
    end
    # println(@sprintf "\t  %-28s%8.2f GB" "Total memory allocated:" Int(round(resH.bytes / 1024^3)))
    flush(stdout)

    # Call garbage collector:
    GC.gc()
    
    # println(typeof(H)) #  Check the type of the Hamiltonian matrix
    # Diagonalize the Hamiltonian:
    # Full-exact diagonalization:
    if lowercase.(metadata.input["FGH"]["method"]) == "exact"
        println("\n\t**** Full Exact diagonalization ****")
        (vals, vecs) = exact_diagonalization(H, metadata.input["FGH"]["Nmax"])
    # Iterative diagonalization:
    elseif lowercase.(metadata.input["FGH"]["method"]) == "iterative"
        println("\n\t** Iterative diagonalization with Lanczos algorithm **")
        flush(stdout)
        (vals, vecs) = Lanczos_algorithm(H, 
            metadata.input["FGH"]["Nmax"],
            tol=10.0^(-metadata.input["FGH"]["advanced"]["tol"]),
            maxiter=metadata.input["FGH"]["advanced"]["maxiter"],
            krylovdim=metadata.input["FGH"]["advanced"]["krylovdim"],
            verbosity=metadata.input["FGH"]["advanced"]["verbosity"]
            )
    else
        throw(ArgumentError("Unknown method. Use 'Exact' or 'Iterative'."))
    end
    flush(stdout)  

    # Renormalize the eigenvectors:
    vecs_rnm = renormalize_vectors(vecs, metadata)
    return (vals, vecs_rnm)
end

function save_eigenstates(eigenstates::Array, metadata::MetaData, energies::Array)
    #= Saves eigenstate as NetCDF file =#
    name = "eigenstates.nc"
    isfile(name) && rm(name)
    NCDataset(name, "c") do outfile
        outfile.attrib["title"] = "Eigenstates and eigenenergies from FGH method."
        if metadata.input["dimensions"] == 1
            defDim(outfile, "x", length(metadata.xdim))
            defDim(outfile, "states", size(eigenstates, 2))
            defVar(outfile, "xdim", Float64, ("x", ))
            defVar(outfile, "potential", Float64, ("x", ))
            defVar(outfile, "wfRe", Float64, ("states", "x"))
            defVar(outfile, "wfIm", Float64, ("states", "x"))
            defVar(outfile, "energies", Float64, ("states", ))
            outfile["xdim"][:] = metadata.xdim
            outfile["potential"][:] = metadata.potential .* constants["Eh_to_wn"]
            for (i, eigstate) in enumerate(eachcol(eigenstates))
                outfile["wfRe"][i, :] = real.(eigstate)
                outfile["wfIm"][i, :] = imag.(eigstate)
                outfile["energies"][i] = energies[i]
            end
            outfile["xdim"].attrib["units"] = "Bohr"
            outfile["potential"].attrib["units"] = "wavenumbers"
            outfile["energies"].attrib["units"] = "wavenumbers"
            outfile["wfRe"].attrib["comments"] = "Real part of the wavefunction" 
            outfile["wfIm"].attrib["comments"] = "Imaginary part of the wavefunction"           
        elseif metadata.input["dimensions"] == 2
            defDim(outfile, "x", length(metadata.xdim))
            defDim(outfile, "y", length(metadata.ydim))
            defDim(outfile, "states", size(eigenstates, 3))
            defVar(outfile, "xdim", Float64, ("x", ))
            defVar(outfile, "ydim", Float64, ("y", ))
            defVar(outfile, "potential", Float64, ("x", "y"))
            defVar(outfile, "wfRe", Float64, ("states", "x", "y"))
            defVar(outfile, "wfIm", Float64, ("states", "x", "y"))
            defVar(outfile, "energies", Float64, ("states", ))
            outfile["potential"][:, :] = metadata.potential .* constants["Eh_to_wn"]
            outfile["xdim"][:] = metadata.xdim
            outfile["ydim"][:] = metadata.ydim
            for (i, eigstate) in enumerate(eachslice(eigenstates, dims=3)) 
                outfile["wfRe"][i, :, :] = real.(eigstate)
                outfile["wfIm"][i, :, :] = imag.(eigstate)
                outfile["energies"][i] = energies[i]
            end
            outfile["xdim"].attrib["units"] = "Bohr"
            outfile["ydim"].attrib["units"] = "Bohr"
            outfile["potential"].attrib["units"] = "wavenumbers"
            outfile["energies"].attrib["units"] = "wavenumbers"           
            outfile["wfRe"].attrib["comments"] = "Real part of the wavefunction" 
            outfile["wfIm"].attrib["comments"] = "Imaginary part of the wavefunction"           
        else
            throw(ArgumentError("Dynamics in $(metadata.input["dimensions"]) dimensions is not supported."))
        end
    end
end



function print_hello()
    hello = """

\t#====================================================
\t            Fourier Grid Hamiltonian
\t====================================================#
\t
\t               Don't Panic!


\t Script calculates variationaly eigenstates and
\t eigenenergies of the Hamiltonian operator in
\t Discrete Variable Representation (DVR) using
\t Fourier Grid Hamiltonian (FGH) method.
\t C. C. Marston, G. G. Balint-Kurti, J. Chem. Phys., 1989, 91(6), 3571-3576,
\t   DOI:10.1063/1.456888

"""
   println(hello)
   flush(stdout)
end

function print_input(input::Dict)
    # Print content of the input file:
    println("\t===============> Input parameters <===============")
    println(@sprintf "\t%-28s%10d" "Number of dimensions" input["dimensions"])
    if input["dimensions"] == 1
        println(@sprintf "\t%-28s%10.4f" "Mass" input["mass"])
    elseif input["dimensions"] == 2
        println(@sprintf "\t%-28s%10.4f" "Mass X" input["mass"][1])
        println(@sprintf "\t%-28s%10.4f" "Mass Y" input["mass"][2])
    end
    println(@sprintf "\t%-28s%10s" "Potential file" input["potential"])
    if haskey(input, "kcoup")
        println(@sprintf "\t%-28s%10.2f" "Kinetic coupling" input["kcoup"])
    end
    println("\tFGH parameters:")
    for (key, value) in input["FGH"]
        if key != "advanced"
            println(@sprintf "\t  %-26s%10s" key value)
        else
            for (key, value) in input["FGH"]["advanced"]
                println(@sprintf "\t  %-26s%10s" key value)
            end
        end
    end
end

#============================================
                Main code
============================================#

starttime = now()
println("\n  Started " * Dates.format(starttime, "on dd/mm/yyyy at HH:MM:SS") * "\n")
println("\t**** Running with $(Threads.nthreads()) threads ****\n")

# Print hello message:
print_hello()

# Parse input file and collect all input parameters into metadata object:
metadata = MetaData(inputfile=infile)
metadata = prepare_inp_param(metadata)
print_input(metadata.input)
println("\n\t============> FGH algorithm started! <=============\n")
flush(stdout)

# Calculate eigenstates and energies employing the FGH method:
(vals, vecs) = execute_FGH(metadata)

# Convert energies to wavenumbers:
vals = vals .* constants["Eh_to_wn"]
# Print energies:
println("\n\tCalculated energies in cm⁻¹:")
for val in vals
    eng = Int(round(round(val)))
    println(@sprintf "\t%12.2f" val)
end

# Save eigenstates and energies to NetCDF file:
println("\n\tSaving eigenstates and energies...")
save_eigenstates(vecs, metadata, vals)
println("\tEigenstates and energies saved to `eigenstates.nc` file.")

# End of the program:
endtime = now()
println("\n  Finished " * Dates.format(endtime, "on dd/mm/yyyy at HH:MM:SS") )
println("  Total runtime: " * Dates.format(convert(DateTime, endtime-starttime ), "HH:MM:SS") * "\n")

