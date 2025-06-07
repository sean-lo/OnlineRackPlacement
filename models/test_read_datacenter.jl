using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using DataFrames

include("$(@__DIR__)/read_datacenter.jl")

data_dir = "$(@__DIR__)/../data/contiguousDataCenterNew"

DC = read_datacenter(
    data_dir,
    toppower_capacity = 1e6,
    midpower_capacity = 200000.0,
    lowpower_capacity = 120000.0,
    cooling_capacity = 4e4,
    failtoppower_scale = 4/3,
    failmidpower_scale = 2.0,
    faillowpower_scale = 2.0,
)