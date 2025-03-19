function read_CSVs_from_dir(
    input_dir::String
)
    return Dict{String, DataFrame}(
        replace(
            fn,
            input_dir => "",
            ".csv" => "",
            "/" => "",
            "\\" => "",
        ) => CSV.read(fn, DataFrame)
        for fn in Glob.glob("*.csv", input_dir)
    )
end
