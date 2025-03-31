include("$(@__DIR__)/utils.jl")

"""
Data structure for immutable attributes.

### Attributes

#### Main set sizes
* `room_IDs` - room ID vector (Int)
* `row_IDs` - row ID vector (Int)
* `power_IDs` - power device ID vector (Int)
* `cooling_IDs` - cooling zone ID vector (Int)
* `tilegroup_IDs` - resource profile ID vector (Int)
* `toppower_IDs` - top power device ID vector (Int)

#### Index sets
* `room_rows_map` - dict of row IDs for each room. (room_ID => [row_IDs])
* `row_room_map` - dict of room IDs for each row. (row_ID => roomID)
* `tilegroup_tiles_map` - dict of tile IDs for each tile group. (resourceProfileID => [tile_IDs])
* `tile_tilegroup_map` - dict of tile group IDs for each tile. (tile_ID => resourceProfileID)
* `tile_position_map` - dict of position IDs for each tile. (tile_ID => posID)
* `row_tilegroups_map` - dict of tile group IDs for each row. (row_ID => [resourceProfileIDs])
* `tilegroup_row_map` - dict of row IDs for each tile group. (resourceProfileID => row_ID)
* `cooling_tiles_map` - dict of tile IDs for each cooling zone. (coolingZoneID => [tile_IDs])
* `cooling_tilegroups_map` - dict of tile group IDs for each cooling zone. (coolingZoneID => [resourceProfileIDs])
* `power_parent_map` - dict of parent IDs for each power device ID, 0 if none (pd => pd's parent)
* `power_children_map` - dict of child power device IDs for each power device. (pd => [child_pd_IDs])
* `power_tilegroups_map` - dict of tile group IDs served by each power device. (pd => [resourceProfileIDs])
* `room_tilegroups_map` - dict of tile group IDs for each room. (room_ID => [resourceProfileIDs])
* `room_toppower_map` - dict of top power device IDs for each room. (room_ID => [toppower_IDs])
* `toppower_room_map` - dict of room IDs for each top power device. (toppower_ID => room_ID)
* `room_power_map` - dict of power device IDs for each room. (room_ID => [power_IDs])
* `power_room_map` - dict of room IDs for each power device. (power_ID => room_ID)

#### Parameters
* `tilegroup_space_capacity` - dict of tilegroup space capacities
* `power_capacity` - dict of PD power capacities
* `failpower_capacity` - dict PD failover power capacities when root power device fails
* `power_balanced_capacity` - dict of balanced power capacity for each room. (room_ID => Float64)
* `cooling_capacity` - dict of CZ capacities
"""
struct DataCenter
    room_IDs::Vector{Int} #
    row_IDs::Vector{Int} #
    power_IDs::Vector{Int} #
    cooling_IDs::Vector{Int} #
    tilegroup_IDs::Vector{Int} #
    toppower_IDs::Vector{Int} #
    # topology
    room_rows_map::Dict{Int, Vector{Int}} #
    row_room_map::Dict{Int, Int} #
    row_tilegroups_map::Dict{Int, Vector{Int}} # 
    tilegroup_row_map::Dict{Int, Int}
    cooling_tilegroups_map::Dict{Int, Vector{Int}} #
    tilegroup_cooling_map::Dict{Int, Int}
    power_parent_map::Dict{Int, Int}
    power_children_map::Dict{Int, Vector{Int}}
    power_tilegroups_map::Dict{Int, Vector{Int}} #
    power_descendants_map::Dict{Int, Vector{Int}} #
    room_tilegroups_map::Dict{Int, Vector{Int}} #
    room_toppower_map::Dict{Int, Vector{Int}} #
    toppower_room_map::Dict{Int, Int}
    room_power_map::Dict{Int, Vector{Int}}
    power_room_map::Dict{Int, Int}

    # capacities - no active demands
    tilegroup_space_capacity::Dict{Int, Float64} # 
    power_capacity::Dict{Int, Float64} #
    failpower_capacity::Dict{Int, Float64} # 
    power_balanced_capacity::Dict{Int, Float64} #
    cooling_capacity::Dict{Int, Float64} #
end

function read_datacenter(
    input_dir::String,
)

    data = read_CSVs_from_dir(input_dir)
    used_keys = ["powerHierarchy", "czTiles", "resourceProfiles", "tiles", "objectCapacities"]

    missing_keys = [
        key for key in used_keys
            if !haskey(data, key)
    ]
    if !isempty(missing_keys)
        error("Missing required data: $(join(missing_keys, ", "))\n$(joinpath(pwd(), input_dir))")
    end
    data = Dict(key => data[key] for key in used_keys)

    for (df_name, fields) in [
        ("powerHierarchy", ["parentPowerDeviceID", "powerDeviceID"]),
        ("czTiles", ["coolingZoneID", "tileID"]),
        ("resourceProfiles", ["resourceProfileID", "tileID"]),
        ("tiles", ["roomID", "rowID"]),
        ("objectCapacities", ["objectType", "objectID", "capacity"]),
    ]
        for field in fields
            if !(field in names(data[df_name]))
                error("Missing required field: $field in $df_name")
            end
        end
    end

    power_IDs = setdiff(
        unique(vcat(
            data["powerHierarchy"][:, :parentPowerDeviceID], 
            data["powerHierarchy"][:, :powerDeviceID]
        )), 
        [0]
    ) |> sort
    cooling_IDs = unique(data["czTiles"][:, :coolingZoneID]) |> sort
    tilegroup_IDs = unique(data["resourceProfiles"][:, :resourceProfileID]) |> sort
    room_IDs = unique(data["tiles"][:, :roomID]) |> sort
    row_IDs = unique(data["tiles"][:, :rowID]) |> sort
    toppower_IDs = unique(data["powerHierarchy"][data["powerHierarchy"][!, :parentPowerDeviceID] .== 0, :powerDeviceID]) |> sort

    # Mappings
    room_rows_map = (
        data["tiles"]
        |> x -> groupby(x, :roomID)
        |> x -> Dict(
            y[!, :roomID][1] => unique(collect(y[!, :rowID]))
            for y in x
        )
    )
    row_room_map = Dict(data["tiles"][!, :rowID] .=> data["tiles"][!, :roomID])
    row_tiles_map = (
        data["tiles"]
        |> x -> groupby(x, :rowID)
        |> x -> Dict(
            y[!, :rowID][1] => unique(collect(y[!, :tileID]))
            for y in x
        )
    )
    tile_row_map = Dict(data["tiles"][!, :tileID] .=> data["tiles"][!, :rowID])
    tilegroup_tiles_map = (
        data["resourceProfiles"]
        |> x -> groupby(x, :resourceProfileID)
        |> x -> Dict(
            y[!, :resourceProfileID][1] => collect(y[!, :tileID])
            for y in x
        )
    )
    tile_tilegroup_map = Dict(data["resourceProfiles"][!, :tileID] .=> data["resourceProfiles"][!, :resourceProfileID])
    row_tilegroups_map = Dict(
        x => unique([
            tile_tilegroup_map[y]
            for y in row_tiles_map[x]
        ])
        for x in row_IDs
    )
    tilegroup_row_map = Dict(
        x => unique([
            tile_row_map[y]
            for y in tilegroup_tiles_map[x]
        ])
        for x in tilegroup_IDs 
    )
    @assert all(length(v) == 1 for v in values(tilegroup_row_map))
    tilegroup_row_map = Dict(x => v[1] for (x, v) in tilegroup_row_map)

    cooling_tiles_map =(
        data["czTiles"]
        |> x -> groupby(x, :coolingZoneID)
        |> x -> Dict(
            y[!, :coolingZoneID][1] => collect(y[!, :tileID])
            for y in x
        )
    )
    cooling_tilegroups_map = Dict(
        x => unique([
            tile_tilegroup_map[y]
            for y in cooling_tiles_map[x]
        ])
        for x in cooling_IDs
    )
    tilegroup_cooling_map = Dict(
        j => c
        for c in cooling_IDs 
            for j in cooling_tilegroups_map[c]
    )

    power_parent_map = Dict(data["powerHierarchy"][!, :powerDeviceID] .=> data["powerHierarchy"][!, :parentPowerDeviceID])
    power_tilegroups_map = Dict(x => Int[] for x in power_IDs)
    for row in eachrow(data["resourceProfiles"])
        tilegroup_ID = row.resourceProfileID
        for pd_ID in [row.powerDeviceID1, row.powerDeviceID2]
            pd_current = pd_ID
            while true
                push!(power_tilegroups_map[pd_current], tilegroup_ID)
                pd_current = power_parent_map[pd_current]
                if pd_current == 0
                    break
                end
            end
        end
    end
    power_tilegroups_map = Dict(x => unique(v) for (x, v) in power_tilegroups_map)

    power_parent_map = Dict(data["powerHierarchy"][!, :powerDeviceID] .=> data["powerHierarchy"][!, :parentPowerDeviceID])
    power_children_map = Dict(
        x => Int[
            data["powerHierarchy"][!, :powerDeviceID][y]
            for y in eachindex(data["powerHierarchy"][!, :parentPowerDeviceID])
            if data["powerHierarchy"][!, :parentPowerDeviceID][y] == x
        ]
        for x in power_IDs
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

    room_tilegroups_map = Dict(
        m => unique([
            j
            for r in room_rows_map[m]
                for j in row_tilegroups_map[r]
        ])
        for m in room_IDs
    )
    room_toppower_map = Dict(
        m => [
            p 
            for p in toppower_IDs
                if length(intersect(
                    power_tilegroups_map[p],
                    room_tilegroups_map[m]
                )) > 0
        ]
        for m in room_IDs
    )
    toppower_room_map = Dict(p => m for (m, p_list) in room_toppower_map for p in p_list)
    room_power_map = Dict(
        m => [
            p 
            for p in power_IDs
                if length(intersect(
                    power_tilegroups_map[p],
                    room_tilegroups_map[m]
                )) > 0
        ]
        for m in room_IDs
    )
    power_room_map = Dict(p => m for (m, p_list) in room_power_map for p in p_list)

    # Capacities
    tilegroup_space_capacity = Dict(j => length(tilegroup_tiles_map[j]) for j in tilegroup_IDs)
    row_capacity = Dict(r => 20 for r in row_IDs)
    power_capacity = (
        data["objectCapacities"]
        |> x -> filter(r -> r[:objectType] == "pd", x)
        |> x -> Dict(
            x[!, :objectID] .=> x[!, :capacity]
        )
    )
    failpower_capacity = (
        data["objectCapacities"]
        |> x -> filter(r -> r[:objectType] == "pdFailover", x)
        |> x -> Dict(
            x[!, :objectID] .=> x[!, :capacity]
        )
    )
    power_balanced_capacity = Dict(
        m => (
            (sum(power_capacity[p] for p in room_toppower_map[m]) * 2) 
            / (length(room_toppower_map[m]) * (length(room_toppower_map[m]) - 1))
        )
        for m in room_IDs
    )

    cooling_capacity = (
        data["objectCapacities"]
        |> x -> filter(r -> r[:objectType] == "cz", x)
        |> x -> Dict(
            x[!, :objectID] .=> x[!, :capacity]
        )
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