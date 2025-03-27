using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using DataFrames

include("$(@__DIR__)/read_datacenter.jl")

data_dir = "$(@__DIR__)/../data/contiguousDataCenterNew"

DC = read_datacenter(data_dir)
