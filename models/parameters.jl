using Parameters
@with_kw struct RackPlacementCoefficients
    placement_reward::Float64 = 200.0
    placement_var_reward::Float64 = 0.0
    discount_factor::Float64 = 0.1
    room_penalty::Float64 = 40.0 # 40, 3, 0 (at 0, 0.2, 1.0)
    room_mult::Float64 = 1.0
    row_penalty::Float64 = 2.0 # 2, 1, 0 (at 0, 0.5, 1.0) 
    row_mult::Float64 = 1.0
    tilegroup_penalty::Float64 = 1.0
    # alpha in paper 
    power_surplus_penalty::Float64 = 1e-5
    # beta in paper
    power_balance_penalty::Float64 = 1e-3
end

@with_kw mutable struct RackPlacementCoefficientsDynamic
    placement_reward::Float64
    placement_var_reward::Float64
    discount_factor::Float64
    room_penalty::Float64
    room_mult::Float64
    room_penalties::Dict{Int, Float64} = Dict{Int, Float64}()
    row_penalty::Float64
    row_mult::Float64
    row_penalties::Dict{Int, Float64} = Dict{Int, Float64}()
    tilegroup_penalty::Float64
    # alpha in paper 
    power_surplus_penalty::Float64
    # beta in paper
    power_balance_penalty::Float64
end

function RackPlacementCoefficientsDynamic(
    RCoeffs::RackPlacementCoefficients,
)
    return RackPlacementCoefficientsDynamic(
        placement_reward = RCoeffs.placement_reward,
        placement_var_reward = RCoeffs.placement_var_reward,
        discount_factor = RCoeffs.discount_factor,
        room_penalty = RCoeffs.room_penalty,
        room_mult = RCoeffs.room_mult,
        row_penalty = RCoeffs.row_penalty,
        row_mult = RCoeffs.row_mult,
        tilegroup_penalty = RCoeffs.tilegroup_penalty,
        power_surplus_penalty = RCoeffs.power_surplus_penalty,
        power_balance_penalty = RCoeffs.power_balance_penalty,
    )
end