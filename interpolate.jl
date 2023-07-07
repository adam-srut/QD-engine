#!/usr/bin/env julia

using Interpolations
using NCDatasets
using YAML
using ArgParse
using Printf
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
        return (x_axis, potential)
    end
end

function read_potential2D(filepath::String; state::Int=3)
    #= Read 2D potential in format:
     x_pos | y_pos | ... | state index | ... =#
    pot = []
    open(filepath) do file
        for line in eachline(file)
            if startswith(line, "#")
                continue
            end
            line = split(line)
            vals = map( x -> parse(Float64, line[x]), [1,2, state])
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
    return (potential, x_dim, y_dim)
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
end


#===================================================
                    Main section
===================================================#

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

if input["dimensions"] == 1
    (xdim, potential) = read_potential(input["potfit"]["potfile"])
    println("""
\t  1D dimensional potential
\t  Potential taken from: $(input["potfit"]["potfile"])
\t  range: [$(xdim[1]), $(xdim[end])]
\t  Number of points before interpolation: $(length(xdim))
\t  Number of points after interpolation: $(input["potfit"]["NPoints"])
""")
    if haskey(input["potfit"], "name")
        fit_potential1D(potential, xdim, input["potfit"]["NPoints"]; name=input["potfit"]["name"])
        println("\t  File $(input["potfit"]["name"]) created.")
    else
        fit_potential1D(potential, xdim, input["potfit"]["NPoints"])
        println("\t  File potential.nc created.")
    end
elseif input["dimensions"] == 2
    (potential, xdim, ydim) = read_potential2D(input["potfit"]["potfile"])
    println("""
\t  2D dimensional potential
\t  Potential taken from: $(input["potfit"]["potfile"])
\t  Xrange: [$(xdim[1]), $(xdim[end])]
\t  yrange: [$(ydim[1]), $(ydim[end])]
\t  Number of points before interpolation: $(length(potential))
\t  Number of points after interpolation: $(input["potfit"]["NPoints"][1]*input["potfit"]["NPoints"][2])
""")
    if haskey(input["potfit"], "name")
        fit_potential2D(potential, xdim, ydim, input["potfit"]["NPoints"][1], input["potfit"]["NPoints"][2]; name=input["potfit"]["name"])
        println("\t  File $(input["potfit"]["name"]) created.")
    else
        fit_potential2D(potential, xdim, ydim, input["potfit"]["NPoints"][1], input["potfit"]["NPoints"][2])
        println("\t  File potential.nc created.")
    end
end


println("\n  Finished successfully " * Dates.format(now(), "on dd/mm/yyyy at HH:MM:SS") * "\n")
