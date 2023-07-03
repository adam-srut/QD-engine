#! /usr/bin/env julia

function compute_spectrum(outdata::OutData;
        maxWn::Number=10000, zpe::Number=0, name::String="spectrum")
    #= Compute spectrum from auto-correlation function.
        cf => Complex array
        timestep => dt*stride in fs
        upperBound => save spectrum up to certain wavenumber =#
    # without window function:
    cf = outdata.CF
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*10) ]
    spectrum = abs.(ifft(cf)) .^ 2
    timestep = outdata.dt*(outdata.step_stride)/constants["fs_to_au"]
    dv = 1/(length(cf)*timestep*1e-15)
    wns = [ zpe + i*dv/constants["c"] for i in 1:length(cf) ]
    spectrum = wns .* spectrum
    spectrum = spectrum/norm(spectrum)
    nyquist = 1/(timestep*1e-15)/2/constants["c"]

    # Add window function:
    hanning(t::Number, T::Number) = (0 <= t <= T) ? ( 1 - cos(2*pi*t/T) )/T : 0
    cf2 = outdata.CF
    cf2 = cf2 .- mean(cf2)
    window = [ hanning(t, length(cf2)) for t in 0:length(cf2)-1 ]
    cf2 = cf2 .* window
    cf2 = [ cf2 ; zeros(length(cf2)*10) ]
    spectrum2 = abs.(ifft(cf2)) .^ 2
    spectrum2 = wns .* spectrum2
    spectrum2 = spectrum2/norm(spectrum2)

    open("$name.dat", "w") do file
        head = @sprintf "# Nyquist freq. = %12.5e [cm⁻¹]\n#%15s%16s%16s\n" nyquist "wn. [cm⁻¹]" "Amp." "Amp. window."
        write(file, head)
        for (i, wn) in enumerate(wns)
            if wn > maxWn
                break
            end
            line = @sprintf "%16.7f%16.7f%16.7f\n" wn spectrum[i] spectrum2[i]
            write(file, line)
        end
    end
end

function compute_energy(dynamics::Dynamics, metadata::MetaData)
    #= Evaluate < ψ | H | ψ > using the Direct Fourier method.
        Energy is returned in Hatrees and composed into kinetic
        and potential energy contributions in Eh. =#
    Vket = dynamics.potential .* dynamics.wf
    Tket = dynamics.PFFT * dynamics.wf
    Tket = fftshift(Tket)
    Tket = dynamics.k_space .* Tket 
    Tket = fftshift(Tket)
    Tket = dynamics.PIFFT * Tket
    if metadata.input["dimensions"] == 1
        wf_norm = abs(trapz(metadata.x_dim, conj(dynamics.wf) .* dynamics.wf))
        Venergy = abs(trapz(metadata.x_dim, conj(dynamics.wf) .* Vket))/wf_norm
        Tenergy = abs(trapz(metadata.x_dim, conj(dynamics.wf) .* Tket))/wf_norm
    elseif metadata.input["dimensions"] == 2
        wf_norm = abs(trapz((metadata.x_dim, metadata.y_dim), conj.(dynamics.wf) .* dynamics.wf))
        Venergy = abs(trapz((metadata.x_dim, metadata.y_dim), conj.(dynamics.wf) .* Vket))/wf_norm
        Tenergy = abs(trapz((metadata.x_dim, metadata.y_dim), conj.(dynamics.wf) .* Tket))/wf_norm
    end
    return (Venergy + Tenergy, Venergy, Tenergy).*constants["Eh_to_wn"]
end

function save_CF(outdata::OutData; filename::String="CF.dat")
    #= Save an array of numbers for each timestep =#
    open(filename, "w") do file
        header = @sprintf "#%15s%16s%16s%16s\n" "time [fs]" "abs" "Re" "Im"
        write(file, header)
        cf = outdata.CF
        for (i, val) in enumerate(cf)
            time = outdata.dt*outdata.step_stride/constants["fs_to_au"]
            time = @sprintf "%16.4f" i*time
            write(file, time)
            val = @sprintf "%16.7f%16.7f%16.7f\n" abs(val) real(val) imag(val)
            write(file, val)
        end
    end
end


function get_eigenfunctions(wf::Array, x_space::Array{Float64}, energies::Array)
    #= Function compute and save vibrational eigenstates.
        Perform FT of ψ(x,t) along time axis.
        Only certain energies will be computed.
        Energies are expected in cm-1
        TODO: add to interface =#

    (N_t, N_x) = size(wf)
    eigenstates = Array{ComplexF64}(undef, length(energies), N_x)
    for (i,energy) in enumerate(energies)
        freq = energy*c*1e-15/fs_to_au
        eigenstate = zeros(N_x)
        for (time,wp) in enumerate(eachrow(wf))
            t = (time-1)*step_stride*dt
            eigenstate = eigenstate + wp*exp(-im*freq*t)
        end
        eigenstates[i,:] = eigenstate
    end
    # Save FT of the wavepacket
    open("eigenstates.txt", "w") do file
        head = @sprintf "#%13s" " "
        head *= join([ @sprintf "%20.2f%8s" e " " for e in energies])
        head *= "\n"
        write(file, head)
        for (i, x) in enumerate(x_space)
            line = @sprintf "%14.4f" x
            for i_state in 1:length(energies)
                val = eigenstates[i_state, i]
                line *= @sprintf "%14.4f%14.4f" real(val) abs(val)
            end
            line *= "\n"
            write(file, line)
        end
    end
end

function save_potential(metadata::MetaData)
    #= Save potential in NetCDF format =#
    isfile("potential.nc") && rm("potential.nc")
    NCDataset("potential.nc", "c") do file
        file.attrib["title"] = "File with potential energy surface used in QD engine"
        if metadata.input["dimensions"] == 1
            defDim(file, "x", length(metadata.x_dim))
            defVar(file, "potential", Float64, ("x",))
            defVar(file, "xVals" , Float64, ("x",))
            file["potential"][:] = metadata.potential
            file["xVals"][:] = metadata.x_dim
        elseif metadata.input["dimensions"] == 2
            defDim(file, "x", length(metadata.x_dim))
            defDim(file, "y", length(metadata.y_dim))
            defVar(file, "potential", Float64, ("x", "y"))
            defVar(file, "xVals", Float64, ("x",))
            defVar(file, "yVals", Float64, ("y",))
            file["potential"][:,:] = metadata.potential
            file["xVals"][:] = metadata.x_dim
            file["yVals"][:] = metadata.y_dim
        end
    end
end

#if abspath(PROGRAM_FILE) == @__FILE__
#    # Heller method for resonance Raman excitation profile
#    # Note: μ_ρ = μ_λ = 1
#    if haskey(input, "resonanceRaman")
#        maxWn = input["resonanceRaman"]["wn_max"]       # Maximal wavenumber in the final spectrum
#        zpe = 1/4/pi*sqrt(k_harm/μ) * fs_to_au*1e15/c   # Compute ZPE
#        cf = Array{Float64}(undef, N_records)           # Initialize correlation function
#        final_state = create_harm_state(1, x_space, x0, k_harm, μ) # Create final scattering states  
#        # Compute cross-correlation function:
#        for (i, wf_t) in enumerate(eachrow(data))
#            cf[i] = abs(dot(final_state, wf_t))
#        end
#        save_vec(cf, "Raman-CF.dat", dt )
#        compute_spectrum(cf, dt*stride/fs_to_au; zpe=zpe, name="resRaman", maxWn=maxWn)
#    end
#end
