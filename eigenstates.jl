#! /usr/bin/env julia

using LinearAlgebra
using NCDatasets, YAML
using ArgParse
using Plots, Plots.PlotMeasures
using Dates
using Trapz

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


#===================================================
                Function section
===================================================#

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

function compute_eigenstate(WF::NCDataset, energy::Number, input::Dict)
    #= Computes eigenstate as:
        ψ(x,y) = ∫ dt ψ(x,y,t) exp( i t Eᵢ/ħ )
        Energy of the eigenstate has to be given in cm-1.
    =# 
    dimensions = length(size(WF["WFRe"])) - 1
    NSteps = size(WF["WFRe"])[end]
    #T = NSteps*input["params"]["dt"]*input["params"]["stride"]
    energy = energy/constants["Eh_to_wn"]
    if dimensions == 1
        eigenstate = zeros(size(WF["WFRe"])[1]) .+ 0im
        for ti in 1:NSteps
            t = ti*input["params"]["stride"]*input["params"]["dt"]
            wf = WF["WFRe"][:, ti] .+ 1im*WF["WFIm"][:, ti]
            eigenstate .+= wf * exp( im*t*energy )
        end
        wfnorm = trapz( WF["Xdim"][:], conj.(eigenstate) .* eigenstate )
        eigenstate = eigenstate/wfnorm
    elseif dimensions == 2
        eigenstate = zeros(size(WF["WFRe"])[1], size(WF["WFRe"])[2]) .+ 0im
        for ti in 1:NSteps
            t = ti*input["params"]["stride"]*input["params"]["dt"]
            wf = WF["WFRe"][:, :, ti] .+ 1im*WF["WFIm"][:, :, ti]
            eigenstate .+= wf * exp( im*t*energy )
        end
        wfnorm = trapz( (WF["Xdim"][:], WF["Ydim"][:]), conj.(eigenstate) .* eigenstate )
        eigenstate = eigenstate/wfnorm
    end 
    eng = Int(round(round(energy*constants["Eh_to_wn"])))
    println("\t    => Eigenstate at energy $eng cm⁻¹ done.")
    return eigenstate
end

function save_eigenstates(eigenstates::Array, WF::NCDataset, energies::Array)
    #= Saves eigenstate as NetCDF file =#
    name = "eigenstates.nc"
    NCDataset(name, "c") do outfile
        if !haskey(WF.dim, "y")
            xdim = WF["Xdim"][:] 
            defDim(outfile, "x", length(xdim))
            defDim(outfile, "states", length(eigenstates))
            defVar(outfile, "Xdim", Float64, ("x", ))
            defVar(outfile, "WFRe", Float64, ("states", "x"))
            defVar(outfile, "WFIm", Float64, ("states", "x"))
            defVar(outfile, "energies", Float64, ("states", ))
            outfile["Xdim"][:] = xdim
            for (i, eigstate) in enumerate(eigenstates)
                outfile["WFRe"][i, :] = real.(eigstate)
                outfile["WFIm"][i, :] = imag.(eigstate)
                outfile["energies"][i] = energies[i]
            end
        else
            xdim = WF["Xdim"][:]
            ydim = WF["Ydim"][:]
            defDim(outfile, "x", length(xdim))
            defDim(outfile, "y", length(ydim))
            defDim(outfile, "states", length(eigenstates))
            defVar(outfile, "Xdim", Float64, ("x", ))
            defVar(outfile, "Ydim", Float64, ("y", ))
            defVar(outfile, "WFRe", Float64, ("states", "x", "y"))
            defVar(outfile, "WFIm", Float64, ("states", "x", "y"))
            defVar(outfile, "energies", Float64, ("states", ))
            for (i, eigstate) in enumerate(eigenstates)
                outfile["WFRe"][i, :, :] = real.(eigstate)
                outfile["WFIm"][i, :, :] = imag.(eigstate)
                outfile["energies"][i] = energies[i]
            end
        end
    end
end

function plot_eigenstates(WF::NCDataset, eigenstates::Array, energies::Array)
    #= Plot provided eigenstates =#
    nstates = length(eigenstates)
    default(fontfamily="computer modern",
        fontfamily_subplot="computer modern",
        framestyle=:box, grid=true,
        legendfontsize=10,
        titlefont = font(10, :black),
        xtickfont = font(10, :black),
        ytickfont = font(10, :black),
        guidefont = font(10, :black),
        tickfontvalign = :vcenter)
    if !haskey(WF.dim, "y")
        xdim = WF["Xdim"][:]    
        for (state, eng) in zip(eigenstates, energies)
            plot(xdim, real.(state), 
                lw=2, color=:black,
                xlabel="X [Bohr]", ylabel="Amplitude", 
                title="Energy: $eng cm⁻¹", legend=false)
            savefig("eigenstate_$(round(eng)).pdf")
        end
    else
        xdim = WF["Xdim"][:]
        ydim = WF["Ydim"][:]
        for (state, eng) in zip(eigenstates, energies)
            contourf(xdim, ydim, real.(state),
                    color=:lightrainbow,
                    xlabel="X [Bohr]", ylabel = "Y [Bohr]",
                    legend=true, levels=10, lw=1.4,
                    framestyle=:box, aspect_ratio=:equal,
                    right_margin=5mm,
                    title="Energy $eng cm⁻¹",
                    size=(500,460)
                    )
            savefig("eigenstate_$(round(eng)).pdf")
        end
    end
end

#===================================================
                Main section
===================================================#

####################################### Info section:
starttime = now()
println("\n\tStarted " * Dates.format(starttime, "on dd/mm/yyyy at HH:MM:SS") * "\n")
println("""
\t#====================================================
\t            Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

""")
########################################

# Read input data
args = parse_commandline()
input = YAML.load_file(args["input"])

WF = NCDataset("WF.nc", "r")

######################################## Info section:
println("""\t============> Spectral method for eigenstates <============\n
\t  Computed as a Fourier transform of a wave packet at a given energy:
\t  ψ(x,y,E) = ∫ ψ(x,y,t) exp(itE/ħ) dt
  
\t  Wave packet evolution taken from `WF.nc` file.""")
########################################


# Calculate eigenfunctions
eig_engs = input["eigstates"]["energies"]
eig_states = [compute_eigenstate(WF, eng, input) for eng in eig_engs]

# Save eigenfunction in NetCDF file and as plots
save_eigenstates(eig_states, WF, eig_engs)
plot_eigenstates(WF, eig_states, eig_engs)

close(WF)

######################################## Info section:
println("\n\t  Results saved to `eigenstates.nc` file.")
endtime = now()
println("\n\tSuccessfully finished " * Dates.format(endtime, "on dd/mm/yyyy at HH:MM:SS") * "\n")
