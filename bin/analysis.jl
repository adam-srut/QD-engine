#! /usr/bin/env julia

function compute_spectrum(outdata::OutData;
        maxWn::Number=10000, zpe::Number=0, name::String="spectrum",
        zff::Int=20, lineshapeWidth::Number=250)
    #= Compute spectrum from auto-correlation function. =#
    # Without the window function:
    cf = outdata.CF
    cf = cf .- mean(cf)
    cf = [ cf ; zeros(length(cf)*zff) ]
    spectrum = abs.(ifft(cf)) 
    timestep = outdata.dt*(outdata.step_stride)/constants["fs_to_au"]
    dv = 1/(length(cf)*timestep*1e-15)
    wns = [ zpe + i*dv/constants["c"] for i in 0:length(cf)-1 ]
    spectrum = wns .* spectrum
    spectrum = spectrum/norm(spectrum)
    nyquist = 1/(timestep*1e-15)/2/constants["c"]

    # Add window function:
    hanning(t::Number, T::Number) = (0 <= t <= T) ? ( 1 - cos(2*pi*t/T) )/T : 0
    cf = outdata.CF
    cf = cf .- mean(cf)
    window = [ hanning(t, length(cf)) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = [ cf ; zeros(length(cf)*zff) ]
    spectrum_hann = abs.(ifft(cf)) 
    spectrum_hann = wns .* spectrum_hann
    spectrum_hann = spectrum_hann/norm(spectrum_hann)

    # Add lineshape function (Gaussian): 
    cf = outdata.CF
    cf = cf .- mean(cf)
    fwhm = lineshapeWidth/timestep
    gauss(t::Number, fwhm::Number) = exp( -t^2/(0.6*fwhm)^2 )
    window = [ gauss(t, fwhm) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = [ cf ; zeros(length(cf)*zff) ]
    spectrum_Gauss = abs.(ifft(cf)) 
    spectrum_Gauss = wns .* spectrum_Gauss
    spectrum_Gauss = spectrum_Gauss/norm(spectrum_Gauss)

    # Add lineshape function (Kubo):
    cf = outdata.CF
    cf = cf .- mean(cf)
    kubo(t::Number, Δ::Number, γ::Number) = exp( -( Δ^2/γ^2 * (γ*t - 1 + exp(-γ*t) ) ) )
    γ = Δ = (lineshapeWidth/timestep)^(-1)
    window = [ kubo(t, Δ, γ) for t in 0:length(cf)-1 ]
    cf = cf .* window
    cf = [ cf ; zeros(length(cf)*zff) ]
    spectrum_Kubo = abs.(ifft(cf)) 
    spectrum_Kubo = wns .* spectrum_Kubo
    spectrum_Kubo = spectrum_Kubo/norm(spectrum_Kubo)

    open("$name.txt", "w") do file
        head = @sprintf "# Nyquist freq. = %12.5e [cm⁻¹]\n#%15s%16s%16s%16s%16s\n" nyquist "wn. [cm⁻¹]" "Amp." "Amp. window." "Gauss LS" "Kubo LS"
        write(file, head)
        for (i, wn) in enumerate(wns)
            if wn > maxWn
                break
            end
            line = @sprintf "%16.7f%16.7f%16.7f%16.7f%16.7f\n" wn spectrum[i] spectrum_hann[i] spectrum_Gauss[i] spectrum_Kubo[i]
            write(file, line)
        end
    end
end

function wp_moments(outdata::OutData, metadata::MetaData)
	#= Calculates 1st and 2nd moments of the WP 
	   According to:
		 mean(x) = <ψ|x|ψ> / <ψ|ψ>
	 	 var(x) = <ψ|x²|ψ> / <ψ|ψ>
       Calculated moments are stored in the output NetCDF file.
	=# 
    # Get the wavepacket data
    wp = outdata.wf
	NSteps = wp.dim["time"]
    # 1D case
	if metadata.input["dimensions"] == 1
        xdim = collect(wp["Xdim"])
        # Calculates normalization factor
        wp0 = wp["WFRe"][:,1] .+ im*wp["WFIm"][:,1]
        wpNorm = real(trapz(xdim, conj.(wp0) .* wp0 ))
        # Initialize output arrays
        WP_center = Array{Float64}(undef, NSteps) # [x](t)
        WP_var = Array{Float64}(undef, NSteps) # [x](t)
        @threads for istep in 1:NSteps
            wpi = wp["WFRe"][:,istep] .+ im*wp["WFIm"][:,istep]
            xmean = trapz(xdim, conj.(wpi) .* xdim .* wpi ) / wpNorm
            xvar = trapz(xdim, conj.(wpi) .* (xdim .^2) .* wpi ) / wpNorm
            WP_center[istep] = real(xmean)
            WP_var[istep] = real(xvar)
        end
        defVar(wp, "x-mean", WP_center, ("time",), attrib = Dict(
            "units" => "Bohr",
            "comments" => "Mean position of the wavepacket, <ψ|x|ψ>")
            )
        defVar(wp, "x-var", WP_var, ("time",), attrib = Dict(
            "units" => "Bohr^2",
            "comments" => "Variance of the wavepacket, <ψ|x²|ψ>")
            )
    # 2D case
    elseif metadata.input["dimensions"] == 2
        xdim = collect(wp["Xdim"])
        ydim = collect(wp["Ydim"])
        # Calculates normalization factor
        wp0 = wp["WFRe"][:,:,1] .+ im*wp["WFIm"][:,:,1]
        wpNorm = real(trapz((xdim, ydim), conj.(wp0) .* wp0 ))
        # Initialize output arrays
        WP_center = Array{Float64}(undef, 2, NSteps) # [x, y](t)
        WP_var = Array{Float64}(undef, 2, NSteps) # [x, y](t)
        # pre-calculate arrays with x or y values only
        xdim_only = repeat(xdim, 1, length(ydim))
        ydim_only = repeat(ydim', length(xdim), 1)
        @threads for istep in 1:NSteps
            wpi = wp["WFRe"][:,:,istep] .+ im*wp["WFIm"][:,:,istep]
            xmean = trapz( (xdim, ydim), conj.(wpi) .* xdim_only .* wpi ) / wpNorm
            ymean = trapz( (xdim, ydim), conj.(wpi) .* ydim_only .* wpi ) / wpNorm
            xvar = trapz( (xdim, ydim), conj.(wpi) .* (xdim_only .^2) .* wpi ) / wpNorm
            yvar = trapz( (xdim, ydim), conj.(wpi) .* (ydim_only .^2) .* wpi ) / wpNorm
            WP_center[:, istep] = real.([xmean, ymean])
            WP_var[:, istep] = real.([xvar, yvar])
    	end
        defVar(wp, "x-mean", WP_center[1,:], ("time",), attrib = Dict(
            "units" => "Bohr",
            "comments" => "Mean position of the wavepacket, <ψ|x|ψ>")
            )
        defVar(wp, "x-var", WP_var[1,:], ("time",), attrib = Dict(
            "units" => "Bohr^2",
            "comments" => "Variance of the wavepacket, <ψ|x²|ψ>")
            )
        defVar(wp, "y-mean", WP_center[2,:], ("time",), attrib = Dict(
            "units" => "Bohr",
            "comments" => "Mean position of the wavepacket, <ψ|y|ψ>")
            )
        defVar(wp, "y-var", WP_var[2,:], ("time",), attrib = Dict(
            "units" => "Bohr^2",
            "comments" => "Variance of the wavepacket, <ψ|y²|ψ>")
            )
    end
	#return (WP_center, WP_var)
end

function compute_energy(dynamics::Dynamics, metadata::MetaData)
    #= Evaluate < ψ | H | ψ > using the Direct Fourier method.
        Energy is returned in Hatrees and composed into kinetic
        and potential energy contributions in Eh. =#
    Vket = dynamics.potential .* dynamics.wf    # V | ψ >
    Tket = dynamics.PFFT * dynamics.wf          # FT[| ψ >]
    Tket = dynamics.k_space .* Tket             # T | k >
    Tket = dynamics.PIFFT * Tket                # FT⁻¹[T | k >]
    # Calculate energy expectation value using trapezoidal rule:
    if metadata.input["dimensions"] == 1
        wf_norm = abs(trapz(metadata.xdim, conj(dynamics.wf) .* dynamics.wf))
        Venergy = abs(trapz(metadata.xdim, conj(dynamics.wf) .* Vket))/wf_norm
        Tenergy = abs(trapz(metadata.xdim, conj(dynamics.wf) .* Tket))/wf_norm
    elseif metadata.input["dimensions"] == 2
        wf_norm = abs(trapz((metadata.xdim, metadata.ydim), conj.(dynamics.wf) .* dynamics.wf))
        Venergy = abs(trapz((metadata.xdim, metadata.ydim), conj.(dynamics.wf) .* Vket))/wf_norm
        Tenergy = abs(trapz((metadata.xdim, metadata.ydim), conj.(dynamics.wf) .* Tket))/wf_norm
    end
    return (Venergy + Tenergy, Venergy, Tenergy)
end

function save_CF(outdata::OutData; filename::String="CF.txt")
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

function GP_spectrum()
    open("spectrum.gp", "w") do outfile
        write(outfile, "set title \"Energy spectrum from autocorrelation function.\"\n")
        write(outfile, "set xrange [0.00:10000]\n")
        write(outfile, "set xlabel 'Energy [cm^-1]'\n")
        write(outfile, "set ylabel 'Intensity'\n")
        write(outfile, "plot \"spectrum.txt\" u 1:2 w l tit \"no window\" lw 1.5 lc rgbcolor \"#5e81b5\", \\
              \"\" u 1:3 w l tit \"Hanning W.\" lw 1.5 lc rgbcolor \"#e19c24\", \\
              \"\" u 1:4 w l tit \"Gauss lineshape\" lw 1.5 lc rgbcolor \"#8fb032\", \\
              \"\" u 1:5 w l tit \"Kubo lineshape\" lw 1.5 lc rgbcolor \"#eb6235\"\n")
        write(outfile, "pause -1")
    end
end

function GP_correlation_function()
    open("CF.gp", "w") do outfile
        write(outfile, "set title \"Autocorrelation function.\"\n")
        write(outfile, "set xlabel 'Time [fs]'\n")
        write(outfile, "set ylabel '<ψ(0)|ψ(t)>'\n")
        write(outfile, "plot \"CF.txt\" u 1:2 w l tit \"|C(t)|\" lw 1.5 lc rgbcolor \"#5e81b5\", \\
              \"\" u 1:3 w l tit \"Re(C(t))\" lw 1.5 lc rgbcolor \"#e19c24\", \\
              \"\" u 1:4 w l tit \"Im(C(t))\" lw 1.5 lc rgbcolor \"#8fb032\"\n")
        write(outfile, "pause -1")
    end
end

