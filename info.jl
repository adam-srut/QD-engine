#! /usr/bin/env julia

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
