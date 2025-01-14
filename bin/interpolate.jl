#!/usr/bin/env julia

using Interpolations
using NCDatasets
using YAML
using ArgParse
using Printf
using Dates
using Plots
using FFTW
using Statistics

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
inputfile = args["input"]

input = YAML.load_file(inputfile)

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
            if startswith(line, "#")
                continue
            end
            line = parse.(Float64, split(line))
            append!(x_axis, line[1])
            append!(potential, line[2])
        end
        perm = sortperm(x_axis)
        potential = potential[perm]
        x_axis = x_axis[perm]
        return (x_axis, potential)
    end
end

function read_potential2D(filepath::String)
    #= Read 2D potential in format:
     x_pos | y_pos | ... | state index | ... =#
    pot = []
    open(filepath) do file
        for line in eachline(file)
            if startswith(line, "#")
                continue
            end
            line = split(line)
            vals = parse.(Float64, line)
            append!(pot, [vals])
        end
    end
    x_dim = sort(unique(map( x -> x[1], pot)))
    N_x = length(x_dim)
    dx = x_dim[2]-x_dim[1]
    y_dim = sort(unique(map( x -> x[2], pot)))
    N_y = length(y_dim)
    dy = y_dim[2]-y_dim[1]
    potential = Array{Float64}(undef, N_x, N_y)
    for line in pot
        x = line[1]
        x_i = round(Int, (x - x_dim[1])/dx) + 1
        y = line[2]
        y_i = round(Int, (y - y_dim[1])/dy) + 1
        potential[x_i, y_i] = line[3]
    end
    # Check for missing values:
    checkMat = fill(false, (N_x, N_y))
    for line in pot
        x = line[1]
        x_i = round(Int, (x - x_dim[1])/dx) + 1
        y = line[2]
        y_i = round(Int, (y - y_dim[1])/dy) + 1
        checkMat[x_i, y_i] = true
    end
    print_warn = false
    open("missing_values.txt", "w") do file
        for ix in 1:N_x
            for iy in 1:N_y
                if !checkMat[ix,iy]
                    x = ((ix-1)*dx + x_dim[1])/1.889725
                    y = ((iy-1)*dy + y_dim[1])/1.889725
                    write(file, @sprintf "%12.4f%12.4f\n" x y)
                    print_warn = true
                end
            end
        end
    end
    if print_warn
        @warn "There are missing values in the potential. Check `missing_values.txt`."
    end
    return (potential, x_dim, y_dim)
end

function smoothing(pot::Array{Float64}, xdim::Array, ydim::Array, FWHM::Float64; padding::Bool=false)
    #=
    Smooth potential with a Gaussian kernel. (method for 2D potentials)
    Function requiers the following args.:
        pot - potential as 2D Array
        xdim - 1D Array of spatial coordinates
        ydim - 1D Array of spatial coordinates
        FWHM - full width at half-maximum for a Gaussian kernel
      Optional:
        padding - Bool; if yes, use linear convolution; if no, use cyclic convolution
      Return:
        2D array with the size of `pot`
    =#
    # Define lengths, dx and dy:
    Nx = length(xdim)
    Ny = length(ydim)
    dx = (xdim[end]-xdim[1])/(Nx-1)
    dy = (ydim[end]-ydim[1])/(Ny-1)
    # Define Gaussian kernel
    kernel = Array{Float64}(undef, Nx, Ny)
    for ix in 1:Nx
        for iy in 1:Ny
            x = dx*(ix-div(Nx,2))
            y = dy*(iy-div(Ny,2))
            kernel[ix,iy] = exp( -((x^2 + y^2)) / ((0.6*FWHM)^2) )
        end
    end
    kernel = kernel/sum(kernel)
    # Substract average from the potential to avoid zero frequency
    absshift = mean(pot)
    pot = pot .- absshift
    # Linear convolution using FT (use zero-padding)
    if padding
        pad_kernel = [ kernel zeros(Nx,Ny-1); zeros(Nx-1,Ny*2-1) ]
        pad_pot = [ pot zeros(Nx,Ny-1); zeros((Nx-1), Ny*2-1)]
        (Nxpad, Nypad) = size(pad_pot)
        conv = ifft( fft(pad_pot) .* fft(pad_kernel))
        conv = real.(conv)
        conv = conv .+ absshift
        return conv[div(Nxpad,4):div(Nxpad,4)*3, div(Nypad,4):div(Nypad,4)*3+1]
    # Cyclic convolution using FT (no padding)
    else
        conv = ifft( fft(pot) .* fft(kernel) )
        conv = circshift(conv, (-div(Nx,2)+1, -div(Ny,2)+1))
        conv = real.(conv)
        conv = conv .+ absshift
        return conv
    end
end

function smoothing(pot::Array, xdim::Array{Float64}, FWHM::Float64; padding::Bool=false)
    #=
    Smooth potential with a Gaussian kernel. (method for 1D potentials)
    Function requiers the following args.:
        pot - potential as 1D Array
        xdim - 1D Array of spatial coordinates
        FWHM - full width at half-maximum for a Gaussian kernel
      Optional:
        padding - Bool; if yes, use linear convolution; if no, use cyclic convolution
      Return:
        1D array with the length of `pot`
    =#
    N = length(xdim)
    dx = (xdim[end]-xdim[1])/(N-1)
    # Define Gaussian kernel
    kernel = [ exp( -(dx*(ix-div(N,2)))^2 / (0.6*FWHM)^2) for ix in 1:N ]
    kernel = kernel/sum(kernel)
    # Substract average from the potential to avoid zero frequencies
    absshift = mean(pot)
    pot = pot .- absshift
    #return kernel
    # Linear convolution using FT (use zero-padding)
    if padding
        pad_pot = [ pot ; zeros(N-1)]
        pad_kernel = [ kernel ; zeros(N-1)]
        Nn = length(pad_pot)
        conv = ifft( fft(pad_pot) .* fft(pad_kernel) )
        conv = real.(conv)
        conv = conv .+ absshift
        return conv[div(Nn,4):div(Nn,4)*3]
    # Cyclic convolution using FT (no padding)
    else
        conv = ifft( fft(pot) .* fft(kernel) )
        conv = circshift(conv, -div(N,2)+1)
        conv = real.(conv)
        conv = conv .+ absshift
        return conv
    end
end

function fit_potential1D(potential::Array{Float64}, xdim::Array{Float64}, NPoints::Int;
        name::String="potential")
    #= Interpolates the provided potential with the cubic spline. 
        Saves result to a NetCDF file. =#
    if ! ispow2(NPoints)
        @warn "Number of grid points is not a power of 2."
    end
    # Define old and new range:
    xrange_old = range(start=xdim[1], stop=xdim[end], length=length(xdim))
    xrange = range(start=xdim[1], stop=xdim[end], length=NPoints)
    # Interpolate and create a new grid:
    itp = cubic_spline_interpolation(xrange_old, potential)
    itp_pot = [ itp(x) for x in xrange ]
    # Save resutls:
    endswith(name, ".nc") ? outname = name : outname = name * ".nc"
    NCDataset(outname, "c") do potfile
        potfile.attrib["title"] = "File with potential energy surface for usage in QD engine"
        defDim(potfile, "x", NPoints)
        defVar(potfile, "xdim", collect(xrange), ("x", ))
        defVar(potfile, "potential", itp_pot, ("x", ))
    end
    plot(collect(xrange), itp_pot,
         xlabel="X [Bohr]",
         ylabel="Energy [Hartree]",
         title="Interpolated potential")
    savefig(outname * ".png")
end

function fit_potential2D(potential::Array{Float64}, xdim::Array{Float64}, ydim::Array{Float64}, 
        NPointsX::Int, NPointsY::Int; name::String="potential")
    #= Interpolates the provided potential with the cubic spline.
        Saves result to a NetCDF file. =#
    if ! ispow2(NPointsX) || ! ispow2(NPointsY)
        @warn "Number of grid points is not a power of 2."
    end
    # Define old and new range:
    xrange_old = range(start=xdim[1], stop=xdim[end], length=length(xdim))
    yrange_old = range(start=ydim[1], stop=ydim[end], length=length(ydim))
    xrange = range(start=xdim[1], stop=xdim[end], length=NPointsX)
    yrange = range(start=ydim[1], stop=ydim[end], length=NPointsY)
    # Interpolate and create a new grid:
    itp = cubic_spline_interpolation((xrange_old, yrange_old), potential)
    itp_pot = vcat([ [ itp(x, y) for x in xrange] for y in yrange ]'...)
    # Check for succesfull interpolation:
    if any(isnan.(itp_pot))
        println("\t" * "*"^20 * "WARNING" * "*"^20)
        println("\t  !!!Interpolation failed!!!\n\t  Check for missing values or discontinuities in $(input["potfit"]["potfile"]).\n")
    end
    # Save resutls:
    endswith(name, ".nc") ? outname = name : outname = name * ".nc"
    NCDataset(outname, "c") do potfile
        potfile.attrib["title"] = "File with potential energy surface for usage in QD engine"
        defDim(potfile, "x", NPointsX)
        defDim(potfile, "y", NPointsY)
        defVar(potfile, "xdim", collect(xrange), ("x", ))
        defVar(potfile, "ydim", collect(yrange), ("y", ))
        defVar(potfile, "potential", itp_pot, ("x", "y"))
    end
    # Plot results:
    contour(collect(xrange), collect(yrange), itp_pot,
            xlabel="X [Bohr]",
            ylabel="Y [Bohr]",
            title="Interpolated potential",
            color=:lighttemperaturemap,
            fill=true,
            aspect_ratio=:equal,
            levels=40,
            size=(600,600))
    savefig(outname * ".png")
end


#===================================================
                    Main section
===================================================#

println("\n  Started " * Dates.format(now(), "on dd/mm/yyyy at HH:MM:SS") * "\n")

hello = """

\t#====================================================
\t            Quantum Dynamics Engine
\t====================================================#
\t
\t               Don't Panic!

   """
println(hello)
println("\t=======> Potential fitting <========\n")
println("\t  Using cubic splines")


if ! haskey(input, "potfit")
    throw(ArgumentError("potfit keyword not found in the input file!"))
end

if haskey(input["potfit"], "FWHM")
    FWHM = input["potfit"]["FWHM"]
else
    throw(ArgumentError("FWHM keyword not found in `potfit` block!
    Please provide the FWHM of Gaussian kernel used for smoothing the potentails (in Borhs).
    Optimal value should be around step-size of the scan."))
end

if input["dimensions"] == 1
    (xdim, potential) = read_potential(input["potfit"]["potfile"])
    potential = smoothing(potential, xdim, FWHM)

########################## INFO
    println("""
\t  1D dimensional potential
\t  Potential taken from: $(input["potfit"]["potfile"])
\t  range: [$(xdim[1]), $(xdim[end])]
\t  Potential smoothed using Gaussian kernel with FWHM = $(FWHM) Bohrs before interpolation.
\t  Number of points before interpolation: $(length(xdim))
\t  Number of points after interpolation: $(input["potfit"]["NPoints"])
""")
##########################
    
    if haskey(input["potfit"], "name")
        fit_potential1D(potential, xdim, input["potfit"]["NPoints"]; name=input["potfit"]["name"])
        println("\t  File $(input["potfit"]["name"]) created.")
    else
        fit_potential1D(potential, xdim, input["potfit"]["NPoints"])
        println("\t  File potential.nc created.")
    end
elseif input["dimensions"] == 2
    (potential, xdim, ydim) = read_potential2D(input["potfit"]["potfile"])
    potential  = smoothing(potential, xdim, ydim, FWHM)

##########################  INFO
    println("""
\t  2D dimensional potential
\t  Potential taken from: $(input["potfit"]["potfile"])
\t  X-range: [$(xdim[1]), $(xdim[end])]
\t  Y-range: [$(ydim[1]), $(ydim[end])]
\t  Potential smoothed using Gaussian kernel with FWHM = $(FWHM) Bohrs before interpolation.
\t  Number of points before interpolation: $(length(potential))
\t  Number of points after interpolation: $(input["potfit"]["NPoints"][1]*input["potfit"]["NPoints"][2])
""")
##########################

    if haskey(input["potfit"], "name")
        fit_potential2D(potential, xdim, ydim, input["potfit"]["NPoints"][1], input["potfit"]["NPoints"][2]; name=input["potfit"]["name"])
        println("\t  File $(input["potfit"]["name"]) created.")
    else
        fit_potential2D(potential, xdim, ydim, input["potfit"]["NPoints"][1], input["potfit"]["NPoints"][2])
        println("\t  File potential.nc created.")
    end
end


println("\n  Finished successfully " * Dates.format(now(), "on dd/mm/yyyy at HH:MM:SS") * "\n")
