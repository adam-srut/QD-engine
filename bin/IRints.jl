#!/usr/bin/env julia

using NCDatasets
using ArgParse
using LinearAlgebra
using Trapz
using Printf
using YAML

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
input = YAML.load_file(infile)

#===================================================
                Basic constants
===================================================#

constants = Dict(
    "wn_to_auFreq" => 7.2516327788e-7,  # cm^-1 to a.u.^-1
    "c_au" => 137.0359991,              # Speed of light in a.u.
    "c" => 29979245800,                 # Speed of light in cm/s
    "fs_to_au" => 41.341373335,
    "ang_to_bohr" => 1.889726125,
    "μ_to_me" => 1822.88849,            # Atomic mass unit in masses of an electron
    "e" => 1.60218e-19,                 # e⁻ charge
    "h" => 6.62607e-34,                 # Planck constant
    "kB" => 3.16681e-6,                 # Boltzmann constant in a.u.
    "Eh_to_wn" => 219474.631
   )

#==============================================
            Function section
==============================================#

function read_dipole(file::String)
    #= Read a dipole moment component in NetCDF format.
        Assume same formatting as in files with potential. =#
    NCDataset(file, "r") do potfile
        if length(potfile.dim) == 1
            xdim = potfile["xdim"][:]
            potential = potfile["potential"][:]
            return potential
        elseif length(potfile.dim) == 2
            xdim = potfile["xdim"][:]
            ydim = potfile["ydim"][:]
            potential = potfile["potential"][:,:]
            return potential
        else
            throw(DimensionMismatch("Wrong number of dimensions in the $filepath. Expected 1 or 2, got $(length(potfile.dim))."))
        end
    end
end

function read_eigenstates(file::String)
    #= Read eigenstates from a NetCDF file. =#
    NCDataset(file, "r") do file
        energies = file["energies"][:]
        if haskey(file.dim, "y")
            eigstates = file["WFRe"][:,:,:] .+ im*file["WFIm"][:,:,:]
            xdim = file["Xdim"][:]
            ydim = file["Ydim"][:]
            for state in 1:file.dim["states"]
                N = sqrt(trapz((xdim,ydim), eigstates[state,:,:] .* conj.(eigstates[state,:,:]) ))
                eigstates[state,:,:] .= eigstates[state,:,:]/N
            end
            return (eigstates, energies, xdim, ydim)
        else
            eigstates = file["WFRe"][:,:] .+ im*file["WFIm"][:,:]
            xdim = file["Xdim"][:]
            for state in 1:file.dim["states"]
                N = sqrt(trapz(xdim, eigstates[state,:] .* conj.(eigstates[state,:]) ))
                eigstates[state,:] .= eigstates[state,:]/N
            end
            return (eigstates, energies, xdim)
        end
    end
end

function compute_trdip(states::Array, energies::Array{Float64}, dips::Array,
        xdim::Array{Float64}, ydim::Array{Float64})
    #= Compute intensities for transition from the lowest eigenstates. =#
    trdips = Array{Float64}(undef, length(energies)-1)
    for istate in 2:length(energies)
        dip_comps = Array{Float64}(undef,3)
        for (i,dipole) in enumerate(dips)
            mu = trapz( (xdim, ydim), conj.(states[istate,:,:]) .* dipole .* states[1,:,:])
            mu = abs(mu)
            dip_comps[i] = mu
        end
        trdips[istate-1] = norm(dip_comps)^2
    end
    engs_au = constants["wn_to_auFreq"]*energies
    tr_engs = engs_au[2:end] .- engs_au[1]
    fosc = 4*pi/3 * trdips .* tr_engs
    trdips = trdips * 2.541746
    return (trdips, fosc)
end

function compute_trdip(states::Array, energies::Array{Float64}, dips::Array, xdim::Array{Float64})
    #= Compute intensities for transition from the lowest eigenstates. =#
    trdips = Array{Float64}(undef, length(energies)-1)
    for istate in 2:length(energies)
        dip_comps = Array{Float64}(undef,3)
        for (i,dipole) in enumerate(dips)
            mu = trapz( xdim, conj.(states[istate,:]) .* dipole .* states[1,:])
            mu = abs(mu)
            dip_comps[i] = mu
        end
        trdips[istate-1] = norm(dip_comps)^2
    end
    engs_au = constants["wn_to_auFreq"]*energies
    tr_engs = engs_au[2:end] .- engs_au[1]
    fosc = 4*pi/3 * trdips .* tr_engs
    trdips = trdips * 2.541746 
    return (trdips, fosc)
end

function save_ints(trdips::Array{Float64}, fosc::Array{Float64}, energies::Array{Float64})
    #= Save transition dipole moments and transition energies =#
    open("IR_intensities.txt", "w") do file
        head = @sprintf "#%15s%16s%16s%16s\n" "transition" "energy [cm-1]" "|μ|^2 [Debye]" "f_osc"
        write(file, head)
        print(head)
        for istate in 1:length(trdips) 
            line = @sprintf "%16s%16.2f%16.4e%16.4e\n" "0 → $istate" energies[istate] trdips[istate] fosc[istate] 
            write(file, line)
            print(line)
        end
    end
end


#==============================================
                Main code
==============================================#

println("""

\t#====================================================
\t            Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

\t  ********* Calculation of IR spectra ************


   """)

if input["dimensions"] == 1
    (eigenstates, energies, xdim) = read_eigenstates(input["irspectrum"]["states"])
    (dipx, dipy, dipz) = [read_dipole(dipfile) for dipfile in input["irspectrum"]["dips"]]
    (trdips, fosc) = compute_trdip(eigenstates, energies, [dipx, dipy, dipz], xdim)
elseif input["dimensions"] == 2
    (eigenstates, energies, xdim, ydim) = read_eigenstates(input["irspectrum"]["states"])
    (dipx, dipy, dipz) = [read_dipole(dipfile) for dipfile in input["irspectrum"]["dips"]]
    (trdips, fosc) = compute_trdip(eigenstates, energies, [dipx, dipy, dipz], xdim, ydim)
end

engs = energies[2:end] .- energies[1]
save_ints(trdips, fosc, engs)


