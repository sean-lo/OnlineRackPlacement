using Pkg
Pkg.activate("$(@__DIR__)/../..")
println(Pkg.status())

include("$(@__DIR__)/../../models/experiment.jl")

const GRB_ENV = Gurobi.Env()

function run_instance(
    args_df, row_index,
    ;
    write::Bool = true,
    result_dir::String = "$(@__DIR__)/logs/$row_index",
    time_limit_sec_per_iteration = 300.0,
)

    (
        run_ind, 
        datacenter_dir, distr_dir, demand_fp,
        method, nummethod,
        use_batching, batch_size,
        online_objectives, discount_factor, 
        S, seed,
    ) = args_df[row_index, :]
    datacenter_dir = String(datacenter_dir)
    distr_dir = String(distr_dir)
    demand_fp = String(demand_fp)
    method = String(method)

    mkpath(result_dir)

    results = run_experiment(
        datacenter_dir,
        distr_dir,
        demand_fp,
        result_dir,
        ;
        write = write,
        env = GRB_ENV,
        strategy = method,
        use_batching = use_batching,
        batch_size = batch_size,
        discount_factor = discount_factor,
        time_limit_sec_per_iteration = time_limit_sec_per_iteration,
        online_objectives = online_objectives,
        seed = seed,
        MIPGap = 1e-4,
    )

    records = [(
        run_ind = run_ind, 
        datacenter_dir = datacenter_dir, 
        distr_dir = distr_dir, 
        demand_fp = demand_fp,
        method = method, 
        nummethod = nummethod,
        online_objectives = online_objectives, 
        discount_factor = discount_factor, 
        S = S, 
        seed = seed,
        total_time = results["time_taken"],
        optimality_gap_mean = results["optimality_gap_mean"],
        demands_placed = results["demands_placed"],
    )]
    write && CSV.write("$(@__DIR__)/results/$row_index.csv", DataFrame(records))
    return
end



# test_args_df = CSV.read("$(@__DIR__)/test_args.csv", DataFrame)
# args_df = CSV.read("$(@__DIR__)/args.csv", DataFrame)
# task_index = parse(Int, ARGS[1]) + 1
# n_tasks = parse(Int, ARGS[2])

# for row_index in 1:1:size(test_args_df, 1)
#     run_instance(
#         test_args_df, row_index,
#         ;
#         write = false,
#         time_limit_sec_per_iteration = 10.0,
#     )
# end

undone_indexes = [
    row_index for row_index in 1:nrow(args_df) 
    if !isfile("$(@__DIR__)/results/$row_index.csv")
]
# for row_index in undone_indexes[task_index:n_tasks:length(undone_indexes)]
for row_index in undone_indexes
    if isfile("$(@__DIR__)/results/$row_index.csv")
        continue
    end
    # sleep(5 * rand())
    println("$row_index")
    run_instance(
        args_df, row_index,
        ;
        write = true,
        time_limit_sec_per_iteration = 300.0,
    )
end
