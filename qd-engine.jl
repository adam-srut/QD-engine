#! /usr/bin/env -S julia --project=/work/qd-engine/project_1dqd

#using Threads
using LinearAlgebra, FFTW
using Statistics
using JSON
using Printf


#===================================================
                Basic constants
===================================================#

wn_to_au = 5.29177210903e-9
c_au = 137.0359991				# Speed of light in a.u.
c = 29979245800                 # Speed of light in cm/s
fs_to_au = 41.341373335
ang_to_bohr = 1.889726125
μ_to_me = 1822.88849        # Atomic mass unit in masses of an electron


#===================================================
                Function section
===================================================#

function read_potential(filepath::String)
    #= Potential is given as two columns:
        position | potential energy =#
    open(filepath) do file
        x_axis = Array{Float64}(undef,0)
        potential = Array{Float64}(undef,0)
        for line in eachline(file)
            if startswith("#", line)
                continue
            end
            line = parse.(Float64, split(line))
            append!(x_axis, line[1])
            append!(potential, line[2])
        end
        return (x_axis, potential)
    end
end



function propagate(potential::Array{Float64}, p::Float64,
        x_space::Array, dt::Float64, wf::Array)
    #= Function for a propagation of a wavepacket in one dimension
        using the Direct Fourier Method.
        All input variables are in atomic units.
        potential => Array of values of potential energy
        p => constant for propagation
        x_space => Array of spatial coordinates
        dt => time increment 
        wf => wavefunction at time t (Complex array)
        NOTE: ADD PREPARED FFT=#
    
    N = length(x_space)
    i_k = -N/2:N/2-1        # Assuming symmetric x range
    wf = fft(wf)
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*p*dt/2) * i_k.^2 )
    wf = fftshift(wf)
    wf = ifft(wf)
    wf = wf .* exp.( -(im*potential*dt) )
    wf = fft(wf)
    wf = fftshift(wf)
    wf = wf .* exp.( -(im*p*dt/2) * i_k.^2 )
    wf = fftshift(wf)
    wf = ifft(wf)
    return wf 
end

function save_data(data::Array, x_space::Array, filename::String)
    open(filename, "w") do file
        header = "# time [fs] Probability amplitude [Bohrs]---->\n"
        write(file, header)
        write(file, " "^12)
        line = map(x -> @sprintf("%12.4f", x), x_space)
        for field in line
            write(file, field)
        end
        write(file, "\n")
        for (i, dat) in enumerate(eachrow(data))
            time = @sprintf "%12.4f" i*stride/fs_to_au
            write(file, time)
            line = map(x -> @sprintf("%12.7f", x), dat)
            for field in line
                write(file, field)
            end
            write(file, "\n")
        end
    end
end

function save_vec(cf::Array, filename)
    open(filename, "w") do file
        header = "# time [fs] val\n"
        write(file, header)
        for (i, val) in enumerate(cf)
            time = @sprintf "%12.4f" i*stride/fs_to_au
            write(file, time)
            val = @sprintf "%12.7f\n" val
            write(file, val)
        end
    end
end

function compute_spectrum(cf::Array, timestep::Number, name::String)
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*10) ]
    spectrum = abs.(fft(cf)) .^ 2
    spectrum = spectrum/norm(spectrum)
    dv = 1/(length(cf)*timestep*1e-15)
    c = 29979245800
    wns = [ i*dv/c for i in 0:length(cf)-1 ]

    open("$name.dat", "w") do file
        for (wn, val) in zip(wns, spectrum)
            if wn > 4000
                break
            end
            line = @sprintf "%12.7f%12.7f\n" wn val
            write(file, line)
        end
    end
end

#===================================================
                Prepare dynamics 
===================================================#

# Read input:
input = JSON.parsefile("input.json")

(x_space, potential) = read_potential(input["potential"])

μ = input["mass"] * μ_to_me
k_harm = input["k_harm"]
x0 = input["initpos"]
wf0 = exp.( -(1/2*sqrt(μ*k_harm) * (x_space .- x0).^2 ))
wf0 = wf0/norm(wf0)
p = (1/2/μ)*(1/(x_space[end] - x_space[1]))^2        # Constant for kinetic energy propagation
dt = input["dt"]
stride = input["stride"]
tmax = input["tmax"]

# Initialize output:
N_records = fld(tmax, stride)
data = Array{Float64}(undef, N_records, length(x_space))
cf = Array{Float64}(undef, N_records)


#===================================================
                Execute dynamics
===================================================#

c = 1
t = 0
wf = wf0 

while t < tmax
    global wf
    global t, c
    wf  = propagate(potential, p, x_space, dt, wf)
    t += dt

    if mod(c, stride) == 0
        i_rec = Int(c/stride)
        data[i_rec, :] = Float64.(wf .* conj(wf))
        cf[i_rec] = norm(dot(wf0, conj(wf)))
    end
    c += 1
end


save_data(data, x_space, "PAmp.dat")
save_vec(cf, "corrF.dat")
compute_spectrum(cf, dt*stride/fs_to_au, "pos-spectrum")

