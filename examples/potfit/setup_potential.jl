#! /usr/bin/env julia

using Printf
using LinearAlgebra

#================================
        Function Section
================================#

function harmonic_pot(μ::Number, nu::Number, x0::Number, x_range)
    #= Prepare harmonic potential.
        μ  - mass in amu
        nu - frequency in cm-1
        x0 - offset in Angstroms
        x_range - range in Angstroms =#
    h_to_wn = 219474.631
    ang_to_bohr = 1.889726125
    wn_to_auFreq = 7.2516327788e-7
    μ_to_me = 1822.88849
    nu = nu*wn_to_auFreq
    μ = μ*μ_to_me
    k = μ*(nu*2*pi)^2
    x0 = x0*ang_to_bohr
    pots = [1/2*k*(x*ang_to_bohr-x0)^2 for x in x_range]
    return pots
end

function morse_pot(μ::Number, nu::Number, x0::Number, x_range)
    #= Prepare harmonic potential.
        μ  - mass in amu
        nu - frequency in cm-1
        x0 - offset in Angstroms
        x_range - range in Angstroms =#
    h_to_wn = 219474.631
    ang_to_bohr = 1.889726125
    wn_to_auFreq = 7.2516327788e-7
    μ_to_me = 1822.88849
    nu = nu*wn_to_auFreq
    μ = μ*μ_to_me
    k = μ*(nu*2*pi)^2
    x0 = x0*ang_to_bohr
    De = 50*2*pi*nu
    a = sqrt(k/2/De)
    pots = [De*(1-exp(-a*(x*ang_to_bohr-x0)))^2 for x in x_range]
    return pots
end

function save_potential(x_range, gs_pot, name)
    # Save computed potentials 
    ang_to_bohr = 1.889726125
    open("$name.txt", "w") do file
        for (x, e) in zip(x_range, gs_pot)
            line = @sprintf "%14.7f%14.7f\n" x*ang_to_bohr e
            write(file, line)
        end
    end
end

#===================================
            Main body
===================================#

# GS potential
x_range = -0.5:0.001:0.5 # in Angstroms
μ = 14.006
nu = 1000
x0 = 0 

potential = harmonic_pot(μ, nu, x0, x_range)
save_potential(x_range, potential, "GS_pot")

# Ex potential
x_range = -0.5:0.001:0.5 # in Angstroms
μ = 14.006
nu = 500
x0 = 0.1

potential = morse_pot(μ, nu, x0, x_range) .+ 0.02
save_potential(x_range, potential, "Ex_pot")


