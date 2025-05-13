#!/usr/bin/env julia

#==================================================
**************************************************
****        Variational Solution of           ****
****  Time-independent Schrödinger Equation   ****
**************************************************
**** Created at TU Darmstadt in Krewald Group.****
**************************************************
**** Author: Adam Šrut                        ****
**************************************************
**************************************************
**** The script is a standalone part of       ****
****   QD-Engine package.                     ****
****   Finds eigenstates and energies of      ****
****   Hamiltoninan in Discrete Value         **** 
****   Representation.                        ****
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

# Variational method parameters:
variational:
    Nmax: 5               # Number of eigenvalues and eigenstates to calculate
    method: "DynamicFourier"     # Method for matrix diagonalization (default: "DynamicFourier")
    # advanced:                  # Advanced options 
    #     precision: "Double"    # Precision of the matrix elements ("Double" or "Single")
    #     krylovdim: 100         # Dimension of the Krylov subspace
    #     tol: 12                # Requested accuracy
    #     maxiter: 100           # Maximum number of iterations
    #     verbosity: 1           # Verbosity level (1-only warnings; 3-info after every iteration) 

# 
# Other possible methods are: "Iterative" and "Exact" 
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
# Keep FFTW serial, FT is pre-planned.
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
    return Tx
end

function construct_T(k_space::Union{Array, Matrix}, metadata::MetaData)
    #= Construct the kinetic energy operator =#
    N = length(k_space)
    # Initialize the kinetic energy operator 
    if lowercase.(metadata.input["variational"]["advanced"]["precision"]) == "single"
        T = zeros(ComplexF32, N,N)
        k_space = Float32.(k_space) 
    elseif lowercase.(metadata.input["variational"]["advanced"]["precision"]) == "double"
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
        copyto!(view(T, :, icol), Tcol)
    end
    return T
end

function construct_H(metadata::MetaData)
    #= Construct the Hamiltonian operator =#
    k_space = metadata.k_space
    pot = metadata.potential
    # Convert potential to single-precision if requested:
    if lowercase.(metadata.input["variational"]["advanced"]["precision"]) == "single"
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

function make_linear_map(metadata::MetaData)
    #= Define the action of the Hamiltonian matrix as a linear map.
        This avoids storing the full matrix in memory.
    =#
    # Plan forward and inverse FFT:
    pfft = plan_fft(metadata.k_space)
    pifft = plan_ifft(metadata.k_space)
    # Define the linear map:
    return function(state_vector::Vector)
        # Reshape the state vector to match the 2D grid:
        if metadata.input["dimensions"] == 2
            state_vector = reshape(state_vector, (length(metadata.xdim), length(metadata.ydim)))
        end
        # Apply the potential energy operator:
        Vket = metadata.potential .* state_vector
        # Apply the kinetic energy operator in k-space:
        Tket = pfft * state_vector          # forward FT
        Tket = metadata.k_space .* Tket     # apply the kinetic energy operator
        Tket = pifft * Tket                 # inverse FT
        # Add the kinetic and potential energy operators:
        state_vector = Tket .+ Vket
        state_vector = vec(state_vector) # flatten the state vector
        return state_vector
    end 
end

function exact_diagonalization(H::Union{Hermitian, Matrix}, nmax::Int)
    #= Exact diagonalization of the Hamiltonian operator 
        Calculates all eigenvalues and eigenvectors, but returns only `nmax`. =#
    # Diagonalize the Hamiltonian:
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

function Lanczos_algorithm(H::Union{Hermitian, Matrix, Function}, nmax::Int, guess::Array{ComplexF64};
        tol::Float64=KrylovDefaults.tol, 
        maxiter::Int=KrylovDefaults.maxiter, 
        krylovdim::Int=100,
        verbosity::Int=1) 
    #= Iterative Lanczos algorithm for diagonalization of the Hamiltonian operator
        using KrylovKit package =#
    result = @timed eigsolve(H,
        guess,                  # initial guess for the eigenvector
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
    #println(@sprintf "\t  %-32s%10.2f GB" "Total memory allocated" result.bytes / 1024^3) 
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
    #= Prepare input parameters for the variational algorithm =#
    advancedParams = Dict(
            "precision" => "Double",  # Precision of the calculation
            "tol" => 12,              # Tolerance for iterative methods
            "maxiter" => 100,         # Maximum number of iterations
            "krylovdim" => 100,       # Krylov subspace dimension
            "verbosity" => 1          # Verbosity level
        )
    # Check if method is provided:
    if !haskey(metadata.input["variational"], "method")
        metadata.input["variational"]["method"] = "DynamicFourier"        
    end
    # Check if the user provided advanced parameters:
    if haskey(metadata.input["variational"], "advanced")
        # Merge default parameters with user-defined parameters
        for (key, value) in metadata.input["variational"]["advanced"]
            advancedParams[key] = value
        end
    end
    # Merge advanced parameters with default parameters:
    metadata.input["variational"]["advanced"] = advancedParams
    return metadata
end

function execute_variational(metadata::MetaData)
    #= Execute the variational algorithm =#
    
    # Dynamic Fourier Method (avoid constructing of the full Hamiltonian matrix): 
    if lowercase.(metadata.input["variational"]["method"]) == "dynamicfourier"
        println("\n\tDynamic Fourier method: ")
        println("\t  The full Hamiltonian matrix will *not* be constructed.")
        println("\t  Actions of kinetic and potential energy operators are calculated directly.")
        println("\n\t** Iterative diagonalization with Lanczos algorithm **")
        flush(stdout)
        # Create a linear map of the Hamiltonian:
        Hmap = make_linear_map(metadata)
        # Diagonalize the Hamiltonian:
        (vals, vecs) = Lanczos_algorithm(Hmap, 
            metadata.input["variational"]["Nmax"],
            rand(ComplexF64, length(metadata.k_space)),
            tol=10.0^(-metadata.input["variational"]["advanced"]["tol"]),
            maxiter=metadata.input["variational"]["advanced"]["maxiter"],
            krylovdim=metadata.input["variational"]["advanced"]["krylovdim"],
            verbosity=metadata.input["variational"]["advanced"]["verbosity"]
            )
        # Renormalize the eigenvectors:
        vecs_rnm = renormalize_vectors(vecs, metadata)
        # Return the eigenvalues and eigenvectors:
        return (vals, vecs_rnm)
    end
    
    # Prepare Hamiltonian:
    println("\tConstructing Hamiltonian...")
    flush(stdout)
    resH = @timed construct_H(metadata)
    H = resH.value
    # Print Hamiltonian info:
    println(@sprintf "\t  %-28s%8d × %d" "Hamiltonian matrix size:" size(H,1) size(H,2))
    println(@sprintf "\t  %-28s%8d s" "Time for construction:" resH.time)
    if lowercase.(metadata.input["variational"]["advanced"]["precision"]) == "single"
        println(@sprintf "\t  %-28s%8.2f GB" "Size in memory:" (size(H,1)^2*8 / 1024^3))
    else
        println(@sprintf "\t  %-28s%8.2f GB" "Size in memory:" (size(H,1)^2*16 / 1024^3))
    end
    # println(@sprintf "\t  %-28s%8.2f GB" "Total memory allocated:" Int(round(resH.bytes / 1024^3)))
    flush(stdout)

    # Call garbage collector:
    GC.gc()
    
    # Diagonalize the Hamiltonian:
    # Full-exact diagonalization:
    if lowercase.(metadata.input["variational"]["method"]) == "exact"
        println("\n\t**** Full Exact diagonalization ****")
        (vals, vecs) = exact_diagonalization(H, metadata.input["variational"]["Nmax"])
    # Iterative diagonalization:
    elseif lowercase.(metadata.input["variational"]["method"]) == "iterative"
        println("\n\t** Iterative diagonalization with Lanczos algorithm **")
        flush(stdout)
        (vals, vecs) = Lanczos_algorithm(H, 
            metadata.input["variational"]["Nmax"],
            rand(ComplexF64, size(H, 1)),
            tol=10.0^(-metadata.input["variational"]["advanced"]["tol"]),
            maxiter=metadata.input["variational"]["advanced"]["maxiter"],
            krylovdim=metadata.input["variational"]["advanced"]["krylovdim"],
            verbosity=metadata.input["variational"]["advanced"]["verbosity"]
            )
    else
        throw(ArgumentError("Unknown method. Use 'Exact', 'Iterative' or 'DynamicFourier'."))
    end
    flush(stdout)  

    # Renormalize the eigenvectors:
    vecs_rnm = renormalize_vectors(vecs, metadata)
    return (vals, vecs_rnm)
end

function save_eigenstates(eigenstates::Array, metadata::MetaData, energies::Array; 
    name::String="eigenstates.nc")
    #= Saves eigenstate as NetCDF file =#
    isfile(name) && rm(name) # Remove an existing file
    NCDataset(name, "c") do outfile
        outfile.attrib["title"] = "Eigenstates and eigenenergies from variational calculation."
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

#===================================================
                Print functions
===================================================#
function print_hello()
    hello = """

\t#====================================================
\t     Variational Solution of Time-Independent
\t       Schrödinger Equation on a Grid
\t====================================================#
\t
\t               Don't Panic!


\t Script finds eigenstates and enenergies of the
\t Hamiltonian operator in  Discrete Variable 
\t Representation (DVR) using Dynamic Fourier method.
\t Kosloff & Kosloff J. Chem. Phys., 79(4), 1983, DOI:10.1063/1.445959

\t Fourier Grid Hamiltonian method is also available (but obsolete).
\t C. C. Marston, G. G. Balint-Kurti, J. Chem. Phys., 1989, 91(6), 3571-3576,
\t   DOI:10.1063/1.456888

"""
   println(hello)
   flush(stdout)
end

function print_input(input::Dict)
    # Print content of the input file:
    println("\t===============> Input parameters <===============")
    println(@sprintf "\t%-28s%14d" "Number of dimensions" input["dimensions"])
    if input["dimensions"] == 1
        println(@sprintf "\t%-28s%14.4f" "Mass" input["mass"])
    elseif input["dimensions"] == 2
        println(@sprintf "\t%-28s%14.4f" "Mass X" input["mass"][1])
        println(@sprintf "\t%-28s%14.4f" "Mass Y" input["mass"][2])
    end
    println(@sprintf "\t%-28s%14s" "Potential file" input["potential"])
    if haskey(input, "kcoup")
        println(@sprintf "\t%-28s%14.2f" "Kinetic coupling" input["kcoup"])
    end
    println("\tvariational parameters:")
    for (key, value) in input["variational"]
        if key != "advanced"
            println(@sprintf "\t  %-26s%14s" key value)
        elseif lowercase.(input["variational"]["method"]) != "exact"
            for (key, value) in input["variational"]["advanced"]
                println(@sprintf "\t  %-26s%14s" key value)
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
println("\n\t============> Algorithm started! <=============\n")
flush(stdout)

# Calculate eigenstates and energies employing the variational method:
(vals, vecs) = execute_variational(metadata)

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

