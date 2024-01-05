#!/usr/bin/env julia

using NCDatasets
using ArgParse
using LinearAlgebra
using Trapz
using Printf

#=================================================
            Parse arguments
=================================================#

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "file1"
            arg_type=String
            required=true
        "file2"
            arg_type=String
            required=true
    end
    return parse_args(s)
end

args = parse_commandline()


#==============================================
            Function section
==============================================#

function compute_ovls(states1::String, states2::String)
    #= Computed overlaps between states in provided NC files =#
    NCDataset(states1, "r") do file1
        NStates1 = file1.dim["states"]
        NCDataset(states2, "r") do file2
            NStates2 = file2.dim["states"]
            ovl_mat = Array{Float64}(undef, NStates1, NStates2)
            if !haskey(file2.dim, "y")
                for s1 in 1:NStates1, s2 in s1:NStates2
                    wf1 = file1["WFRe"][s1,:] .+ im*file1["WFIm"][s1,:]
                    wf2 = file2["WFRe"][s2,:] .+ im*file2["WFIm"][s2,:]
                    xdim = file1["Xdim"][:]
                    norm1 = sqrt(abs(trapz(xdim, wf1 .* conj.(wf1))))
                    norm2 = sqrt(abs(trapz(xdim, wf2 .* conj.(wf2))))
                    ovl  = abs(trapz(xdim, wf1/norm1 .* conj.(wf2/norm2)))
                    ovl_mat[s1,s2] = ovl
                end
            else
                for s1 in 1:NStates1, s2 in s1:NStates2
                    wf1 = file1["WFRe"][s1,:,:] + im*file1["WFIm"][s1,:,:]
                    wf2 = file2["WFRe"][s2,:,:] + im*file2["WFIm"][s2,:,:]
                    xdim = file1["Xdim"][:]
                    ydim = file1["Ydim"][:]
                    norm1 = sqrt(abs(trapz((xdim, ydim), wf1 .* conj.(wf1))))
                    norm2 = sqrt(abs(trapz((xdim, ydim), wf2 .* conj.(wf2))))
                    ovl  = abs(trapz((xdim, ydim), wf1/norm1 .* conj.(wf2/norm2)))
                    ovl_mat[s1,s2] = ovl
                end
            end
            ovl_mat = Hermitian(ovl_mat)
            return ovl_mat
        end
    end
end

function print_overlap(ovls::Hermitian)
    #= Print the overlap matrix as upper triangular =#
    (NStates1, NStates2) = size(ovls)
    println("""\n\t==========> Overlap matrix <===========\n""")
    print("\t")
    [ print(@sprintf "%10d" j) for j in 1:NStates2 ]
    print("\n")
    for i in 1:NStates1, j in i:NStates2
        if i == j
            print("\t$i" * " "^(10*(i-1)))
        end
        print(@sprintf "%10.5f" ovls[i,j])
        if j == NStates2
            print("\n\n")
        end
    end
    print("\n")
end

#==============================================
                Main code
==============================================#

ovl = compute_ovls(args["file1"], args["file2"])
print_overlap(ovl)

