using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using DataFrames

include("$(@__DIR__)/build_datacenter.jl")

data_dir = "$(@__DIR__)/../data/contiguousDataCenterNew"

DC = build_datacenter(data_dir)
