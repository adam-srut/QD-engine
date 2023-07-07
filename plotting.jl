#!/usr/bin/env julia

using ArgParse
using Plots
using YAML
using NCDatasets
using LinearAlgebra
gr()

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


function plot_time(times::Array{Int}, wfout::NCDataset, input::Dict, potential::NCDataset)
    #=  =#
    colors = [ "#5e81b5", "#e19c24", "#8fb032", "#eb6235", "#8778b3", "#c56e1a",
              "#5d9ec7", "#ffbf00", "#a5609d", "#929600", "#e95536", "#6685d9", 
              "#f89f13", "#bc5b80", "#47b66d" ]
    if length(times) > length(colors)
        error("Too many frames to plot. Max is: $(length(colors))")
    end
    if input["dimensions"] == 1
        x_space = wfout["Xdim"][:]
        plot(size=(650,500),
             titlefont = font(10, :black),
             xtickfont = font(10, :black),
             ytickfont = font(10, :black),
             guidefont = font(10, :black),
             labelfont = font(10, :gray),
             foreground_color_legend = :gray,
             legend_background_color = :white,
             xlabel = "position [Bohrs]",
             ylabel = "Probability amplitude"
            )
         plot!(twinx(), x_space, potential["potential"][:],
              color=:black,
              line=(2, :dash),
              legend=false,
              xtickfont = font(10, :black),
              ytickfont = font(10, :black),
              ylabel = "Energy [Eh]"
             )
         for (i, time) in enumerate(times)
            timefs = round(time*input["params"]["dt"]*input["params"]["stride"]/41.3413; digits=2)
            PAmp = wfout["PAmp"][:, time] 
            plot!(x_space, PAmp,
                  color=colors[i],
                  lw=2,
                  label="t = $timefs fs"
                 )
        end
        savefig("./PAmp_evolution.pdf")
     if input["dimensions"] == 2
         x_space = wfout["Xdim"][:]
         y_space = wfout["Ydim"][:]

    end
end

#===================================================
                Main section
===================================================#

wfout = NCDataset("WF.nc", "r")
potential = NCDataset(input["potential"], "r")

t = collect(1:60:400)

plot_time(t, wfout, input, potential)








