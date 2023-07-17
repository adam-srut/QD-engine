#! /usr/bin/env julia

#===================================================
                    INFO section
===================================================#
function print_hello()
    hello = """

\t#====================================================
\t            Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

   """
   println(hello)
   flush(stdout)
end

function print_init(metadata::MetaData)
    #= Prints information about the input =#
    #println("\n    Basic information:\n")
    println("""\tUsing Direct Fourier Method for propagation, see:
            \t  Kosloff & Kosloff J. Chem. Phys., 79(4), 1983.\n
            \tfor the split operator formalism see:
            \t  Feit, Fleck & Steiger, J. Comput. Phys., 1982, 47, 412–433.\n""")
    #println("\t" * "*"^60)
    if metadata.input["dimensions"] == 1
        println("\n\t========> 1-dimensional dynamics <========")
        println(@sprintf "\t Particle mass:%8.4f amu" metadata.input["mass"] )
        if haskey(metadata.input["initWF"], "fromfile")
            println("\t Reading ψ(0,x) from: $(metadata.input["initWF"]["fromfile"])")
        else
            println("\t Using Gaussian WP at t=0:")
            println(@sprintf "\t\t%-20s%12.3f Bohrs" "centered at:" metadata.input["initWF"]["initpos"] )
            println(@sprintf "\t\t%-20s%12.3f cm⁻¹" "harm. freq.:" metadata.input["initWF"]["freq"] )
            fwhm = 2*sqrt(2*log(2))/((metadata.input["initWF"]["freq"]*constants["wn_to_auFreq"]*2*pi) * metadata.input["mass"]*constants["μ_to_me"])
            println(@sprintf "\t\t%-20s%12.3f Bohrs\n" "FWHM:" fwhm )
        end
        println(@sprintf "\t %-20s%12.2f a.u." "Timestep:" metadata.input["params"]["dt"] )
        println(@sprintf "\t %-20s%12.f a.u." "tₘₐₓ:" metadata.input["params"]["Nsteps"]*metadata.input["params"]["dt"] )
        println(@sprintf "\t %-20s%12.f fs" "tₘₐₓ:" metadata.input["params"]["Nsteps"]*metadata.input["params"]["dt"]/constants["fs_to_au"] )
        println(@sprintf "\t %-20s%12d steps" "stride:"  metadata.input["params"]["stride"]  )
        println("\tInformation about potential:")
        println(@sprintf "\t %-20s%12.2e Bohrs" "Grid spacing:" metadata.xdim[2]-metadata.xdim[1] )
        println(@sprintf "\t %-20s%12d" "Grid points:" length(metadata.xdim) )
        println(@sprintf "\t %-20s%12.4f a.u." "Maximal momentum:" 1/2/metadata.xdim[2]-metadata.xdim[1] )
        println(@sprintf "\n\t %-22s%12s\n" "Potential taken from:" metadata.input["potential"] )
    elseif metadata.input["dimensions"] == 2
        println("\n\t========> 2-dimensional dynamics <======== ")
        println(@sprintf "\t Particle mass 1:%8.4f amu" metadata.input["mass"][1] )
        println(@sprintf "\t Particle mass 2:%8.4f amu" metadata.input["mass"][2] )
        if haskey(metadata.input["initWF"], "fromfile")
            println("\t Reading ψ(0,x,y) from: $(metadata.input["initWF"]["fromfile"])")
        else
            println("\t Using Gaussian WP at t=0:")
            println(@sprintf "\t\t%-20s%12.3f,%11.3f Bohrs" "centered at:" metadata.input["initWF"]["initpos"]... )
            println(@sprintf "\t\t%-20s%12.3f,%11.3f cm⁻¹" "harm. freq.:" metadata.input["initWF"]["freq"]... )
            fwhm1 = 2*sqrt(2*log(2))/((metadata.input["initWF"]["freq"][1]*constants["wn_to_auFreq"]*2*pi) * metadata.input["mass"][1]*constants["μ_to_me"])
            fwhm2 = 2*sqrt(2*log(2))/((metadata.input["initWF"]["freq"][2]*constants["wn_to_auFreq"]*2*pi) * metadata.input["mass"][2]*constants["μ_to_me"])
            println(@sprintf "\t\t%-20s%12.3f,%11.3f Bohrs\n" "FWHM:" fwhm1 fwhm2 )
        end
        println(@sprintf "\t %-20s%12.2f a.u." "Timestep:" metadata.input["params"]["dt"] )
        println(@sprintf "\t %-20s%12.f a.u." "tₘₐₓ:" metadata.input["params"]["Nsteps"]*metadata.input["params"]["dt"] )
        println(@sprintf "\t %-20s%12.f fs" "tₘₐₓ:" metadata.input["params"]["Nsteps"]*metadata.input["params"]["dt"]/constants["fs_to_au"] )
        println(@sprintf "\t %-20s%12d steps" "stride:"  metadata.input["params"]["stride"]  )
        println("\tInformation about potential:")
        println(@sprintf "\t %-20s%12.2e Bohrs" "X-Grid spacing:" metadata.xdim[2]-metadata.xdim[1] )
        println(@sprintf "\t %-20s%12.2e Bohrs" "Y-Grid spacing:" metadata.ydim[2]-metadata.ydim[1] )
        println(@sprintf "\t %-20s%12d" "Grid points:" length(metadata.potential) )
        println(@sprintf "\t %-20s%12.4f,%11.4f a.u." "Maximal momentum:" abs(1/2/metadata.xdim[2]-metadata.xdim[1]) abs(1/2/metadata.ydim[2]-metadata.ydim[1]) )
        println(@sprintf "\n\t%-22s%12s\n" "Potential taken from:" metadata.input["potential"] )
    end
    
    println("\n\t============> Warnings <============")
    println("\t Maximum bandwidth: Δt < π/3/ΔVₘₐₓ:")
    println(@sprintf "\t\t%6.2f <%6.2f" metadata.input["params"]["dt"] pi/3/(maximum(metadata.potential)-minimum(metadata.potential)))
    if metadata.input["params"]["dt"] > pi/3/(maximum(metadata.potential)-minimum(metadata.potential))
        @warn "Aliasing errors are likely to occur. Decrease the time step or check bounds of the potential."
    end
    println("\t Minimal resolution of energy levels: ΔEₘᵢₙ = π/T:")
    println(@sprintf "\t\t%8.4f cm⁻¹" pi/(metadata.input["params"]["Nsteps"]*metadata.input["params"]["dt"])*constants["Eh_to_wn"])
    if haskey(metadata.input, "imagT")
        println("\t Imaginary time propagation requested.\n\t Reference and actual potential need to have the same spatial grid")
    end

    println("\n")

    println("\t" * "*"^60)
    println("\tInitial condition:\n")
    if metadata.input["dimensions"] == 1
        plt_min = Int(round(0.1*length(metadata.xdim)))
        plt_max = Int(round(0.9*length(metadata.xdim)))
        pot_max = max(metadata.potential[plt_min:plt_max]...)
        pot_min = min(metadata.potential[plt_min:plt_max]...)
        wf0 = abs.(dynamics.wf) .^ 2
        wf_max = maximum(wf0)
        initPlot = lineplot(metadata.xdim[plt_min:plt_max], metadata.potential[plt_min:plt_max],
                           canvas=BrailleCanvas,
                           border=:ascii, color=:white, name="potential",
                           xlabel="Distance [Bohrs]", ylabel="Energy [Hartree]",
                           height=10, grid=false, compact=true, blend=false)
        wf_rescale = 0.5*(pot_max - pot_min)/wf_max
        wf_to_plot = wf0 * wf_rescale .+ 1.1*pot_min
        lineplot!(initPlot, metadata.xdim, wf_to_plot, color=:cyan, name="WF(t=0)")
        println(initPlot)
        println("\t" * "*"^60 * "\n\n")
    elseif metadata.input["dimensions"] == 2
        initPlot = contourplot(metadata.xdim, metadata.ydim, metadata.potential,
                               canvas=BrailleCanvas,
                               border=:ascii, colorbar=false,
                               xlabel="X [Bohrs]", ylabel="Y [Bohrs]",
                               height=10, width=20,
                               grid=false, compact=true, blend=false)
        contourplot!(initPlot, metadata.xdim, metadata.ydim, Float64.( conj.(dynamics.wf) .* dynamics.wf),
                    colormap=:Greys_3)
        println(initPlot)
        println("\t" * "*"^60 * "\n\n")
        flush(stdout)
    end
end


function print_run()
    message = """

\t#====================================================
\t              Running dynamics:
\t====================================================#

"""
    println(message)
    flush(stdout)
end

function print_output()
    message = """

\t#====================================================
\t              Analyzing and saving:
\t====================================================#
\t
\t    |ψ(t,x,[y])|² saved in NetCDF format to `WF.nc` file.
\t    Correlation function <ψ(0)|ψ(t)> saved in ASCII format to `CF.txt`.
\t    Spectrum saved in ASCII format to `spectrum.txt`, computed as:
\t       σ(ν) = ν/2/π ∫ <ψ(0)|ψ(t)> exp(i2πνt) dt 
\t       with and without the Hanning window: (1-cos(2πt/T) ∀ t ∊ (0,T), 0 elsewhere.
\t    Gnuplot scripts for spectrum and corr. func.: `CF.gp`, `spectrum.gp` 
"""

    println(message)
    flush(stdout)
end

