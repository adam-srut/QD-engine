#! /usr/bin/env julia
using NCDatasets
using LinearAlgebra, Statistics
using FFTW
using Printf
using YAML
using ArgParse
using Dates

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
                Function section
===================================================#

function compute_spectrum(CF::Array{ComplexF64}, step_stride::Int, dt::Number, Nsteps::Int;
        maxWn::Number=10000, minWN::Number=0, zpe::Number=0, outname::String="spectrum_lineshape",
        zff::Int=20, lineshapeWidth::Number=250, powerspectrum::Bool=false)
    #= Compute spectrum from correlation function.
        Using various window or lineshape functions. =#
    timestep = dt*step_stride/41.341373335
    totT = length(CF)*timestep
    T = length(CF)*(zff+1)
    dv = 1/(T*timestep*1e-15)
    wns = [ -zpe + i*dv/29979245800 for i in 0:T-1 ]

    # Add window function:
    hanning(t::Number, T::Number) = (0 <= t <= T) ? ( 1 - cos(2*pi*t/T) )/T : 0
    cf = copy(CF)
    window = [ hanning(t, length(cf)) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*zff) ]
    powerspectrum ? spectrum_hann = abs.(ifft(cf)).^2 : spectrum_hann = abs.(ifft(cf))
    spectrum_hann = abs.(wns) .* spectrum_hann / totT
    spectrum_hann = spectrum_hann/maximum(spectrum_hann)

    # Add lineshape function (Gaussian): 
    cf = copy(CF)
    fwhm = lineshapeWidth/timestep
    gauss(t::Number, fwhm::Number) = exp( -t^2/(0.6*fwhm)^2 )
    window = [ gauss(t, fwhm) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*zff) ]
    powerspectrum ? spectrum_Gauss = abs.(ifft(cf)).^2 : spectrum_Gauss = abs.(ifft(cf))
    spectrum_Gauss = abs.(wns) .* spectrum_Gauss / totT
    spectrum_Gauss = spectrum_Gauss/maximum(spectrum_Gauss)

    # Add lineshape function (Kubo):
    cf = copy(CF)
    kubo(t::Number, Δ::Number, γ::Number) = exp( -( Δ^2/γ^2 * (γ*t - 1 + exp(-γ*t) ) ) )
    γ = (lineshapeWidth/timestep)^(-1)
    Δ = (lineshapeWidth*4/7/timestep)^(-1)
    window = [ kubo(t, Δ, γ) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*zff) ]
    powerspectrum ? spectrum_Kubo = abs.(ifft(cf)).^2 : spectrum_Kubo = abs.(ifft(cf)) 
    spectrum_Kubo = abs.(wns) .* spectrum_Kubo / totT
    spectrum_Kubo = spectrum_Kubo/maximum(spectrum_Kubo)

    open("$outname.txt", "w") do file
        head = @sprintf "#%15s%16s%16s%16s\n" "wn. [cm⁻¹]" "Hann window." "Gauss LS" "Kubo LS"
        write(file, head)
        for (i, wn) in enumerate(wns)
            if wn > maxWn
                break
            end
            line = @sprintf "%16.7f%16.7e%16.7e%16.7e\n" wn spectrum_hann[i] spectrum_Gauss[i] spectrum_Kubo[i]
            write(file, line)
        end
    end
end

function readCF_NetCDF()
    #= Read correlation function from `WF.nc` file =#
    if !isfile("WF.nc")
        error("File `WF.nc` not found.")
    end
    NCDataset("WF.nc", "r") do wffile
        CF = wffile["CFRe"][:] + im*wffile["CFIm"][:]
        return CF
    end
end

function readCF_ASCII(input::Dict)
    #= Read correlation function from `CF.txt` file =#
    Nrec = Int(fld(input["params"]["Nsteps"], input["params"]["stride"]))
    CF = Array{ComplexF64}(undef, Nrec)
    open("CF.txt") do CFfile
        i = 1
        for line in eachline(CFfile)
            if startswith(line, '#')
                continue
            end
            line = split(line)
            (t, absCF, ReCF, ImCF) = parse.(Float64, line)
            CF[i] = ReCF + 1im*ImCF
            i += 1
        end
    end
    return CF
end

function read_operator(filepath::String)
    #= Read operator for spectra calculation without Condon approximation.
        File is expected to have same formatting as potential.nc =#
    NCDataset(filepath, "r") do file
        if length(file.dim) == 1
            xdim = file["xdim"][:]
            operator = file["potential"][:]
            return operator
        elseif length(file.dim) == 2
            xdim = file["xdim"][:]
            ydim = file["ydim"][:]
            operator = file["potential"][:,:]
            return operator
        else
            throw(DimensionMismatch("Wrong number of dimensions in the $filepath. Expected 1 or 2, got $(length(potfile.dim))."))
        end
    end
end

function read_ZPE()
    #= Read ZPE from `initWF.nc` =#
    if !isfile("initWF.nc")
        error("File `initWF.nc` not found.")
    end
    NCDataset("initWF.nc", "r") do file
        return file["ZPE"][1]
    end
end

function write_GP(outname::String)
    #= Create Gnuplot script for plotting the spectrum =#
    open("$outname.gp", "w") do outfile
        write(outfile, "set title \"Energy spectrum from autocorrelation function.\"\n")
        write(outfile, "set xrange [$(input["spectrum"]["minWn"]):$(input["spectrum"]["maxWn"])]\n")
        write(outfile, "set xlabel 'Energy [cm^-1]'\n")
        write(outfile, "set ylabel 'Intensity'\n")
        write(outfile, "plot \"$outname.txt\" u 1:2 w l tit \"Hanning W.\" lw 1.5 lc rgbcolor \"#5e81b5\", \\
              \"\" u 1:3 w l tit \"Gaussian lineshape\" lw 1.5 lc rgbcolor \"#e19c24\", \\
              \"\" u 1:4 w l tit \"Kubo lineshape\" lw 1.5 lc rgbcolor \"#8fb032\"\n")
        write(outfile, "pause -1\n")
    end
end

function resRaman_corrF(eigstate::Int, input::Dict)
    #= Computes correlation function for resonance Raman
        absorption profile. 
       eigstate => index of an eigenstate in `eigenstates.nc` 
       `WF.nc` has to be present to read the WP evolution =#
    local fstate
    if !isfile("eigenstates.nc")
        error("File `eigenstates.nc` not found.")
    elseif !isfile("WF.nc")
        error("File `WF.nc` not found.")
    end
    # Read the final scattering state:
    NCDataset("eigenstates.nc", "r") do eigfile
        if !haskey(eigfile.dim, "y")
            fstate = eigfile["WFRe"][eigstate, :] .+ 1im*eigfile["WFIm"][eigstate, :]
            fstate = fstate / norm(fstate)
        else
            fstate = eigfile["WFRe"][eigstate, :, :] .+ 1im*eigfile["WFIm"][eigstate, :, :]
            fstate = fstate / norm(fstate)
        end
    end
    # Check for non-Condon
    if haskey(input, "noncondon")
        mu = read_operator(input["noncondon"]["dip"])
    end
    # Read temporal evolution of the WP at the excited state surface:
    Nrec = Int(fld(input["params"]["Nsteps"], input["params"]["stride"]))
    CF = Array{ComplexF64}(undef, Nrec)
    NCDataset("WF.nc", "r") do wffile
        for i in 1:Nrec
            if input["dimensions"] == 1
                wf = wffile["WFRe"][:, i] .+ 1im*wffile["WFIm"][:, i]
            elseif input["dimensions"] == 2
                wf = wffile["WFRe"][:, :, i] .+ 1im*wffile["WFIm"][:, :, i]
            end
            if haskey(input, "noncondon")
                overlap = dot(fstate, mu .* wf)
            else
                overlap = dot(fstate, wf)
            end
            CF[i] = overlap
        end
    end

    # Save correlation function:
    open("cross-CF_$eigstate.txt", "w") do outfile
        header = @sprintf "#%15s%16s%16s%16s\n" "time [fs]" "abs" "Re" "Im"
        write(outfile, header)
        for (i, val) in enumerate(CF)
            time = input["params"]["dt"]*input["params"]["stride"]/41.341373335
            time = @sprintf "%16.4f" i*time
            write(outfile, time)
            val = @sprintf "%16.7e%16.7e%16.7e\n" abs(val) real(val) imag(val)
            write(outfile, val)
        end
    end
    open("cross-CF_$eigstate.gp", "w") do outfile
        write(outfile, "set title \"Cross-correlation function.\"\n")
        write(outfile, "set xlabel 'Time [fs]'\n")
        write(outfile, "set ylabel '<ψ_f|ψ_i(t)>'\n")
        write(outfile, "plot \"cross-CF_$eigstate.txt\" u 1:2 w l tit \"|C(t)|\" lw 1.5 lc rgbcolor \"#5e81b5\", \\
              \"\" u 1:3 w l tit \"Re(C(t))\" lw 1.5 lc rgbcolor \"#e19c24\", \\
              \"\" u 1:4 w l tit \"Im(C(t))\" lw 1.5 lc rgbcolor \"#8fb032\"\n")
        write(outfile, "pause -1")
    end
    return CF
end

#===================================================
                Main section
===================================================#

########################## Info:
starttime = now()
println("\n\tStarted " * Dates.format(starttime, "on dd/mm/yyyy at HH:MM:SS") * "\n")

hello = """

\t#====================================================
\t            Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

   """
   println(hello)

println("\t============> Spectra generation <============\n")
println("\t  Spectrum as a Fourier transform of a correlation function:")
println("\t    σ(ν) = ν/2/π ∫ <ψ(0)|ψ(t)> ⋅ LS(t) exp(i2πνt) dt")
println("\t    where LS(t) is a lineshape function.")
###########################

input = YAML.load_file(infile)

# Vibronic spectrum or energy levels:
if haskey(input, "spectrum")
    println("\n\t ========> Vibronic spectrum <=========")
    println("\t  Using autocorrelation function to compute spectrum.")
    println("\t   --> Vibronic spectrum or energy levels.")
    if haskey(input["spectrum"], "outname")
        outname = input["spectrum"]["outname"]
    else
        outname = "spectrum_lineshape"
    end
    
    if isfile("WF.nc")
        CF = readCF_NetCDF()
        println("\t  Correlation function read from `WF.nc` file.")
    else
        CF = readCF_ASCII(input)
        println("\t  Correlation function read from `CF.txt` file.")
    end

    if input["spectrum"]["ZPE"] == "read"
        zpe = read_ZPE()
        println("\t  Zero-point energy will be read from file `initWF.nc`.")
        println(@sprintf "\t    ZPE = %6.2f cm⁻¹, used for frequency shift in Fourier transform." zpe)
        compute_spectrum(CF, input["params"]["stride"], input["params"]["dt"], input["params"]["Nsteps"];
                         maxWn=input["spectrum"]["maxWn"], minWN=input["spectrum"]["minWn"], 
                         zpe=zpe, lineshapeWidth=input["spectrum"]["linewidth"],
                         outname=outname)
    else
        println(@sprintf "\t    ZPE = %6.2f cm⁻¹, used for frequency shift in Fourier transform." input["spectrum"]["ZPE"])
        compute_spectrum(CF, input["params"]["stride"], input["params"]["dt"], input["params"]["Nsteps"];
                         maxWn=input["spectrum"]["maxWn"], minWN=input["spectrum"]["minWn"],
                         zpe=input["spectrum"]["ZPE"], lineshapeWidth=input["spectrum"]["linewidth"],
                         outname=outname)
    end
    write_GP(outname)
    println("""
\t  Spectra saved to: $outname.txt
\t  Gnuplot script for plotting: $outname.gp
""")
end

# Resonance Raman scattering cross section:
if haskey(input, "Raman")
    println("\n\t ========> Resonance Raman <=========")
    println("\t  Using cross-correlation function to compute spectrum.")
    println("\t   --> Resonance Raman scattering cross section.")
    if input["Raman"]["ZPE"] == "read"
        zpe = read_ZPE()
        println("\t  Zero-point energy will be read from file `initWF.nc`.")
    else
        zpe = input["Raman"]["ZPE"]
    end
    println(@sprintf "\t    ZPE = %6.2f cm⁻¹, used for frequency shift in Fourier transform." zpe)
    println("")
    
    for finalstate in input["Raman"]["finalstate"]
        local outname, CF
        outname = "resRaman_1$finalstate"
        println("\t  Computing cross-correlation function with eigenstate #$finalstate.")
        CF = resRaman_corrF(finalstate, input)
        
        compute_spectrum(CF, input["params"]["stride"], input["params"]["dt"], input["params"]["Nsteps"];
                         maxWn=input["Raman"]["maxWn"], minWN=input["Raman"]["minWn"],
                         zpe=zpe, lineshapeWidth=input["Raman"]["linewidth"],
                         outname=outname, powerspectrum=true) 
        write_GP(outname)
        println("\t  => Spectra saved to: $outname.txt; $outname.gp")
        println("""\n
\t  Heller method for frequency dependent polarizability:
\t    see: Lee & Heller, J. Chem. Phys., 71(12), 1979, 4777–4788
\t    α(i→ f) = ∫ <ψ_f | μ_S exp(-iĤ₂t/ħ) μ_I | ψ_i > dt """)

    end
end

############################ Info:

fwhm = input["spectrum"]["linewidth"]
println("\n\t  Lineshapes functions used to produce spectra:")
println("\t    - Gaussian: exp( t² / (0.6 ⋅ $(input["spectrum"]["linewidth"]) fs)² )")
println(@sprintf "\t    - Kubo:  exp(-( Δ²/γ² * (γ*t - 1 + exp(-γ*t)))); Δ = %8.6f fs⁻¹; γ = %8.6f fs⁻¹" 1/(fwhm*4/7) 1/fwhm )

endtime = now()
println("\n\tSuccessfully finished " * Dates.format(endtime, "on dd/mm/yyyy at HH:MM:SS") * "\n")


