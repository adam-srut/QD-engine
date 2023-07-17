#! /usr/bin/env julia
using NCDatasets
using LinearAlgebra, Statistics
using FFTW
using Printf
using YAML
using ArgParse

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

function compute_spectrum(filename::String, step_stride::Int, dt::Number, Nsteps::Int;
        maxWn::Number=10000, minWN::Number=0, zpe::Number=0, outname::String="spectrum_lineshape",
        zff::Int=20, lineshapeWidth::Number=250)
    local CF
    #= Compute spectrum from auto-correlation function. =#
    NCDataset(filename, "r") do wffile
        @assert Int(fld(Nsteps, step_stride)) == wffile.dim["timeCF"] "Nsteps and stride do not correspond with NR in WF.nc file."
        CF = wffile["CFRe"][:] + im*wffile["CFIm"][:]
    end
    timestep = dt*step_stride/41.341373335
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
    spectrum_hann = abs.(ifft(cf)) 
    spectrum_hann = abs.(wns) .* spectrum_hann
    spectrum_hann = spectrum_hann/maximum(spectrum_hann)

    # Add lineshape function (Gaussian): 
    cf = copy(CF)
    fwhm = lineshapeWidth/timestep
    gauss(t::Number, fwhm::Number) = exp( -t^2/(0.6*fwhm)^2 )
    window = [ gauss(t, fwhm) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*zff) ]
    spectrum_Gauss = abs.(ifft(cf)) 
    spectrum_Gauss = abs.(wns) .* spectrum_Gauss
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
    spectrum_Kubo = abs.(ifft(cf)) 
    spectrum_Kubo = abs.(wns) .* spectrum_Kubo
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

function read_ZPE()
    #= Read ZPE from `initWF.nc` =#
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

#===================================================
                Main section
===================================================#

input = YAML.load_file(infile)

if haskey(input["spectrum"], "outname")
    outname = input["spectrum"]["outname"]
else
    outname = "spectrum_lineshape"
end

if input["spectrum"]["ZPE"] == "read"
    zpe = read_ZPE()
    compute_spectrum("WF.nc", 
                     input["params"]["stride"], input["params"]["dt"], input["params"]["Nsteps"];
                     maxWn=input["spectrum"]["maxWn"], minWN=input["spectrum"]["minWn"], 
                     zpe=zpe, lineshapeWidth=input["spectrum"]["linewidth"],
                     outname=outname)
else
    compute_spectrum("WF.nc",
                     input["params"]["stride"], input["params"]["dt"], input["params"]["Nsteps"];
                     maxWn=input["spectrum"]["maxWn"], minWN=input["spectrum"]["minWn"],
                     zpe=input["spectrum"]["ZPE"], lineshapeWidth=input["spectrum"]["linewidth"],
                     outname=outname)
end

write_GP(outname)

