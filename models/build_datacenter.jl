using Pkg
Pkg.activate("$(@__DIR__)/../")

using CSV
using Glob
using Combinatorics
using DataFrames

include("$(@__DIR__)/read_datacenter.jl")

function build_datacenter(
    ;
    n_rooms::Int = 2,
    n_rows::Int = 36,
    n_1power::Int = 4,
    n_2power::Int = 6,
    n_3power::Int = 3,
    n_cooling::Int = 18,
    n_tilegroups::Int = 72,
    toppower_capacity::Float64 = 1e6,
    midpower_capacity::Float64 = 200000.0,
    lowpower_capacity::Float64 = 120000.0,
    failtoppower_scale::Float64 = 4/3,
    failmidpower_scale::Float64 = 2.0,
    faillowpower_scale::Float64 = 2.0,
    cooling_capacity::Float64 = 1e6,
)

    power_offset1 = n_1power
    power_offset2 = n_1power + (n_1power * n_2power)
    n_power = ((n_3power + 1) * n_2power + 1) * n_1power

    room_IDs = collect(1:n_rooms)
    row_IDs = collect(1:n_rows * n_rooms)
    power_IDs = collect(1:n_power * n_rooms)
    cooling_IDs = collect(1:n_cooling * n_rooms)
    tilegroup_IDs = collect(1:n_tilegroups * n_rooms)
    toppower_IDs = Int[]
    midpower_IDs = Int[]
    lowpower_IDs = Int[]
    for m in room_IDs
        append!(toppower_IDs, collect(
            ((m - 1) * n_power) 
            .+ (1:n_1power)
        ))
        append!(midpower_IDs, collect(
            ((m - 1) * n_power + n_1power) 
            .+ (1:n_2power * n_1power)
        ))
        append!(lowpower_IDs, collect(
            ((m - 1) * n_power + n_2power * n_1power + n_1power) 
            .+ (1:n_3power * n_2power * n_1power)
        ))
    end

    room_rows_map = Dict{Int, Vector{Int}}(
        m => collect(((m - 1) * n_rows) .+ (1:n_rows))
        for m in room_IDs
    )
    row_room_map = Dict{Int, Int}(
        r => m
        for m in room_IDs
            for r in room_rows_map[m]
    )
    
    row_tilegroups_map = Dict{Int, Vector{Int}}()
    for m in room_IDs
        for r in 1:n_rows
            row_ind = (m - 1) * n_rows + r
            row_tilegroups_map[row_ind] = [
                (m - 1) * n_tilegroups + r, 
                (m - 1) * n_tilegroups + r + n_rows, 
            ]
        end
    end
    tilegroup_row_map = Dict{Int, Int}(
        t => r
        for r in row_IDs 
            for t in row_tilegroups_map[r]
    )
    
    cooling_tilegroups_map = Dict{Int, Vector{Int}}()
    for m in room_IDs
        for c in 1:n_cooling
            cool_ind = (m - 1) * n_cooling + c
            cooling_tilegroups_map[cool_ind] = [
                (m - 1) * n_tilegroups + c, 
                (m - 1) * n_tilegroups + n_rows - c + 1,
                (m - 1) * n_tilegroups + n_rows + c,
                (m - 1) * n_tilegroups + 2*n_rows - c + 1,
            ]
        end
    end
    tilegroup_cooling_map = Dict{Int, Int}(
        t => c
        for c in cooling_IDs
            for t in cooling_tilegroups_map[c]
    )
    
    power_children_map = Dict{Int, Vector{Int}}()
    for m in room_IDs
        for p1 in 1:n_1power
            p1_ind = (m - 1) * n_power + p1
            power_children_map[p1_ind] = [
                (m - 1) * n_power + power_offset1 + (p1-1)*n_2power + p2
                for p2 in 1:n_2power
            ]
        end
        for p1 in 1:n_1power, p2 in 1:n_2power
            p2_ind = (m - 1) * n_power + power_offset1 + (p1-1)*n_2power + p2
            power_children_map[p2_ind] = [
                (m - 1) * n_power + power_offset2 + (p1-1)*n_2power*n_3power + (p2-1)*n_3power + p3
                for p3 in 1:n_3power
            ]
        end
        for p1 in 1:n_1power, p2 in 1:n_2power, p3 in 1:n_3power
            p3_ind = (m - 1) * n_power + power_offset2 + (p1-1)*n_2power*n_3power + (p2-1)*n_3power + p3
            power_children_map[p3_ind] = Int[]
        end
    end


    power_parent_map = Dict{Int, Int}(
        p_ => p
        for p in power_IDs
            for p_ in power_children_map[p]
    )
    
    power_descendants_map = Dict{Int, Vector{Int}}()
    for pd_ID in power_IDs
        power_descendants_map[pd_ID] = Int[]
        frontier = Int[pd_ID]
        while length(frontier) > 0
            append!(power_descendants_map[pd_ID], frontier)
            frontier = vcat([
                power_children_map[pd_ID]
                for pd_ID in frontier
            ]...)
        end
    end

    tilegroup_power_info = Dict(
        t => Dict{String, Any}()
        for t in 1:n_tilegroups
    )
    # Power device
    for (i, (p1, p2)) in enumerate(collect(combinations(1:n_1power, 2))) # row
        for (i_, (p1_, p2_)) in enumerate(collect(combinations(1:n_1power, 2))) # col
            t = (i-1) * (n_1power * (n_1power - 1) / 2) + i_
            tilegroup_power_info[t]["p1_tuple"] = (p1, i_, mod((p2 - p1), n_1power))
            tilegroup_power_info[t]["p2_tuple"] = (p2, i_, mod((p1 - p2), n_1power))
            tilegroup_power_info[t + n_rows]["p1_tuple"] = (p1_, i, mod((p2_ - p1_), n_1power))
            tilegroup_power_info[t + n_rows]["p2_tuple"] = (p2_, i, mod((p1_ - p2_), n_1power))
        end
    end
    for t in 1:n_tilegroups
        tilegroup_power_info[t]["p1"] = power_offset2 + (
            (
                (tilegroup_power_info[t]["p1_tuple"][1] - 1) * n_2power 
                + (tilegroup_power_info[t]["p1_tuple"][2] - 1)
            ) * n_3power
            + tilegroup_power_info[t]["p1_tuple"][3]
        )
        tilegroup_power_info[t]["p2"] = power_offset2 + (
            (
                (tilegroup_power_info[t]["p2_tuple"][1] - 1) * n_2power
                + (tilegroup_power_info[t]["p2_tuple"][2] - 1)
            ) * n_3power
            + tilegroup_power_info[t]["p2_tuple"][3]
        )
    end
    

    tilegroup_power_map = Dict{Int, Vector{Int}}()
    for m in room_IDs
        for t in 1:n_tilegroups
            t_ind = (m - 1) * n_tilegroups + t
            tilegroup_power_map[t_ind] = [
                (m - 1) * n_power + tilegroup_power_info[t]["p1"], 
                (m - 1) * n_power + tilegroup_power_info[t]["p2"],
            ]
        end
    end

    power_tilegroups_map = Dict{Int, Vector{Int}}(
        p => [t for t in tilegroup_IDs if p in tilegroup_power_map[t]]
        for p in power_IDs
    )
    for p in midpower_IDs
        power_tilegroups_map[p] = vcat([
            power_tilegroups_map[p_] 
            for p_ in power_children_map[p]
        ]...) |> unique |> sort
    end
    for p in toppower_IDs
        power_tilegroups_map[p] = vcat([
            power_tilegroups_map[p_] 
            for p_ in power_children_map[p]
        ]...) |> unique |> sort
    end

    room_tilegroups_map = Dict{Int, Vector{Int}}(
        m => collect((m - 1) * n_tilegroups .+ (1:n_tilegroups))
        for m in room_IDs
    )
    room_toppower_map = Dict{Int, Vector{Int}}(
        m => collect((m - 1) * n_power .+ (1:n_1power))
        for m in room_IDs
    )
    toppower_room_map = Dict{Int, Int}(
        tp => m
        for m in room_IDs
            for tp in room_toppower_map[m]
    )
    room_power_map = Dict{Int, Vector{Int}}(
        m => collect((m - 1) * n_power .+ (1:n_power))
        for m in room_IDs
    )
    power_room_map = Dict{Int, Int}(
        p => m
        for m in room_IDs
            for p in room_power_map[m]
    )

    tilegroup_space_capacity = Dict(tilegroup_IDs .=> 10)
    row_capacity = Dict(row_IDs .=> 20)
    power_capacity = Dict{Int, Float64}()
    failpower_capacity = Dict{Int, Float64}()
    for p in toppower_IDs
        power_capacity[p] = toppower_capacity
        failpower_capacity[p] = power_capacity[p] * failtoppower_scale
    end
    for p in midpower_IDs
        power_capacity[p] = midpower_capacity
        failpower_capacity[p] = power_capacity[p] * failmidpower_scale
    end
    for p in lowpower_IDs
        power_capacity[p] = lowpower_capacity
        failpower_capacity[p] = power_capacity[p] * faillowpower_scale
    end
    cooling_capacity = Dict(cooling_IDs .=> cooling_capacity)
    power_balanced_capacity = Dict(
        m => (
            (sum(power_capacity[p] for p in room_toppower_map[m]) * 2) 
            / (length(room_toppower_map[m]) * (length(room_toppower_map[m]) - 1))
        )
        for m in room_IDs
    )

    return DataCenter(
        room_IDs, row_IDs, power_IDs, cooling_IDs, tilegroup_IDs, toppower_IDs,
        room_rows_map, row_room_map, 
        row_tilegroups_map, tilegroup_row_map,
        cooling_tilegroups_map, tilegroup_cooling_map,
        power_parent_map, power_children_map,
        power_tilegroups_map, power_descendants_map, 
        room_tilegroups_map,
        room_toppower_map, toppower_room_map, room_power_map, power_room_map,
        tilegroup_space_capacity, 
        power_capacity, failpower_capacity, power_balanced_capacity,
        cooling_capacity, 
    )
end