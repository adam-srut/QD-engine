#! /usr/bin/env julia


function compute_spectrum(cf::Array, timestep::Number;
        maxWn::Number=10000, zpe::Number=0, name::String="spectrum")
    #= Compute spectrum from auto-correlation function.
        cf => Complex array
        timestep => dt*stride in fs
        upperBound => save spectrum up to certain wavenumber
        TODO: add lineshape function =#
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*5) ]
    spectrum = abs.(fft(cf)) .^ 2
    dv = 1/(length(cf)*timestep*1e-15)
    wns = [ zpe + i*dv/constants["c"] for i in 0:length(cf)-1 ]
    spectrum = wns .* spectrum
    spectrum = spectrum/norm(spectrum)
    nyquist = 1/(timestep*1e-15)/2/constants["c"]

    open("$name.dat", "w") do file
        head = @sprintf "# Nyquist freq. = %12.5e [cm⁻¹]\n#%11s%12s\n" nyquist "wn. [cm⁻¹]" "Amplitude"
        write(file, head)
        for (wn, val) in zip(wns, spectrum)
            if wn > maxWn
                break
            end
            line = @sprintf "%12.7f%12.7f\n" wn val
            write(file, line)
        end
    end
end

function compute_energy(wf::Array, potential::Array{Float64}, p::Float64)
    #= Evaluate < ψ | H | ψ > using the Direct Fourier method 
        Requires wavefunction, potential and propagation constant as an input.=#
    N = length(potential)
    i_k = -N/2:N/2-1
    ket = potential .* wf
    ket = fft(ket)
    ket = fftshift(ket)
    ket = p * i_k.^2 .* ket
    ket = fftshift(ket)
    ket = ifft(ket)
    energy = dot(wf,ket)/dot(wf,wf)
    return energy
end

function save_vec(cf::Array, filename::String, dt::Float64)
    #= Save an array of numbers for each timestep =#
    open(filename, "w") do file
        header = "# time [fs] val\n"
        write(file, header)
        for (i, val) in enumerate(cf)
            time = @sprintf "%12.4f" (i-1)*stride*dt/fs_to_au
            write(file, time)
            val = @sprintf "%12.7f\n" val
            write(file, val)
        end
    end
end

function save_data(data::Array, x_space::Array, dt::Float64)
    open("PAmp.dat", "w") do file
        header = "# time [fs] Probability amplitude [Bohrs]---->\n"
        write(file, header)
        write(file, " "^12)
        line = map(x -> @sprintf("%12.4f", x), x_space)
        for field in line
            write(file, field)
        end
        write(file, "\n")
        for (i, dat) in enumerate(eachrow(data))
            time = @sprintf "%12.4f" i*stride*dt/fs_to_au
            write(file, time)
            dat = Float64.( conj(dat) .* dat )
            line = map(x -> @sprintf("%12.7f", x), dat)
            for field in line
                write(file, field)
            end
            write(file, "\n")
        end
    end
    open("PAmp-kspace.dat", "w") do file
        header = "# time [fs] Probability amplitude [momentum]---->\n"
        write(file, header)
        write(file, " "^12)
        map( x -> write(file, @sprintf("%12d", x)), -length(x_space)/2:(length(x_space)/2-1) )
        write(file, "\n")
        for (i, dat) in enumerate(eachrow(data))
            if mod(i,2) == 0
                continue
            end
            time = @sprintf "%12.4f" i*stride*dt/fs_to_au
            write(file, time)
            dat = Array{ComplexF64}(dat)
            k_dat = fftshift(fft(dat))
            k_dat = Float64.( conj(k_dat) .* k_dat )
            map(x -> write(file, @sprintf("%12.7f", x)), k_dat)
            write(file, "\n")
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Heller method for resonance Raman excitation profile
    # Note: μ_ρ = μ_λ = 1
    if haskey(input, "resonanceRaman")
        maxWn = input["resonanceRaman"]["wn_max"]       # Maximal wavenumber in the final spectrum
        zpe = 1/4/pi*sqrt(k_harm/μ) * fs_to_au*1e15/c   # Compute ZPE
        cf = Array{Float64}(undef, N_records)           # Initialize correlation function
        final_state = create_harm_state(1, x_space, x0, k_harm, μ) # Create final scattering states  
        # Compute cross-correlation function:
        for (i, wf_t) in enumerate(eachrow(data))
            cf[i] = abs(dot(final_state, wf_t))
        end
        save_vec(cf, "Raman-CF.dat", dt )
        compute_spectrum(cf, dt*stride/fs_to_au; zpe=zpe, name="resRaman", maxWn=maxWn)
    end
end
