using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, Test
using StatsModels: @formula, term

# Function to load and reshape data
function load_and_reshape_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:size(df,1))
    
    # Reshape Y variables
    dfy = select(df, :bus_id, r"Y\d+", :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long.time = parse.(Int, replace.(string.(dfy_long.variable), r"[^\d]" => ""))
    select!(dfy_long, Not(:variable))

    # Reshape Odometer variables
    dfx = select(df, :bus_id, r"Odo\d+")
    dfx_long = DataFrames.stack(dfx, Not(:bus_id))
    rename!(dfx_long, :value => :Odometer)
    dfx_long.time = parse.(Int, replace.(string.(dfx_long.variable), r"[^\d]" => ""))
    select!(dfx_long, Not(:variable))

    # Join the reshaped dataframes
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    # Ensure correct types
    df_long.Y = Int.(df_long.Y)
    df_long.Odometer = Float64.(df_long.Odometer)
    df_long.RouteUsage = Float64.(df_long.RouteUsage)
    df_long.Branded = Int.(df_long.Branded)
    df_long.time = Int.(df_long.time)
    
    return df_long
end

# Function to estimate flexible logit model
function estimate_flexible_logit(df_long)
    # Create polynomial terms
    df_long.Odometer2 = df_long.Odometer .^ 2
    df_long.RouteUsage2 = df_long.RouteUsage .^ 2
    df_long.time2 = df_long.time .^ 2

    # Specify the formula
    formula = @formula(Y ~ (Odometer + Odometer2 + RouteUsage + RouteUsage2 + Branded + time + time2)^7)
    
    # Fit the model
    model = glm(formula, df_long, Binomial(), LogitLink())
    return model
end

# Function to prepare data parameters
function prepare_data_params(df_long)
    println("Debug: Starting prepare_data_params")
    unique_bus_ids = unique(df_long.bus_id)
    N = length(unique_bus_ids)
    T = length(unique(df_long.time))
    
    zbin = length(unique(df_long.RouteUsage))
    xbin = 10  # You might want to adjust this based on your data
    xval = range(minimum(df_long.Odometer), maximum(df_long.Odometer), length=xbin)
    zval = sort(unique(df_long.RouteUsage))
    
    println("Debug: N=$N, T=$T, zbin=$zbin, xbin=$xbin")

    Xstate = zeros(Int, N, T)
    Zstate = Vector{Int}(undef, N)
    B = Vector{Int}(undef, N)

    for (i, bus) in enumerate(unique_bus_ids)
        bus_data = df_long[df_long.bus_id .== bus, :]
        Xstate[i, :] = min.(searchsortedfirst.(Ref(xval), bus_data.Odometer), xbin)
        Zstate[i] = findfirst(==(bus_data.RouteUsage[1]), zval)
        B[i] = bus_data.Branded[1]
    end
    
    # Create a simple transition matrix (you may need to adjust this based on your model)
    xtran = zeros(Int, zbin * xbin, xbin)
    for i in 1:(zbin * xbin)
        xtran[i, :] = [j == 1 ? xbin : j-1 for j in 1:xbin]
    end

    println("Debug: Xstate size: $(size(Xstate))")
    println("Debug: Zstate size: $(size(Zstate))")
    println("Debug: B size: $(size(B))")
    println("Debug: xtran size: $(size(xtran))")

    println("Debug: Completed prepare_data_params")
    return (N=N, T=T, zbin=zbin, xbin=xbin, xval=collect(xval), zval=zval, 
            Xstate=Xstate, Zstate=Zstate, B=B, xtran=xtran)
end
function prepare_data_params(df_long)
    println("Debug: Starting prepare_data_params")
    unique_bus_ids = unique(df_long.bus_id)
    N = length(unique_bus_ids)
    T = length(unique(df_long.time))
    
    zbin = length(unique(df_long.RouteUsage))
    xbin = 10  # You might want to adjust this based on your data
    xval = range(minimum(df_long.Odometer), maximum(df_long.Odometer), length=xbin)
    zval = sort(unique(df_long.RouteUsage))
    
    println("Debug: N=$N, T=$T, zbin=$zbin, xbin=$xbin")

    Xstate = zeros(Int, N, T)
    Zstate = Vector{Int}(undef, N)
    B = Vector{Int}(undef, N)

    for (i, bus) in enumerate(unique_bus_ids)
        bus_data = df_long[df_long.bus_id .== bus, :]
        Xstate[i, :] = min.(searchsortedfirst.(Ref(xval), bus_data.Odometer), xbin)
        Zstate[i] = findfirst(==(bus_data.RouteUsage[1]), zval)
        B[i] = bus_data.Branded[1]
    end
    
    # Create a simple transition matrix (you may need to adjust this based on your model)
    xtran = zeros(Int, zbin * xbin, xbin)
    for i in 1:(zbin * xbin)
        xtran[i, :] = [j == 1 ? xbin : j-1 for j in 1:xbin]
    end

    println("Debug: Xstate size: $(size(Xstate))")
    println("Debug: Zstate size: $(size(Zstate))")
    println("Debug: B size: $(size(B))")
    println("Debug: xtran size: $(size(xtran))")

    println("Debug: Completed prepare_data_params")
    return (N=N, T=T, zbin=zbin, xbin=xbin, xval=collect(xval), zval=zval, 
            Xstate=Xstate, Zstate=Zstate, B=B, xtran=xtran)
end
# Function to estimate structural parameters
function run_ccp_estimation()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long = load_and_reshape_data(url)
    
    flexible_logit_model = estimate_flexible_logit(df_long)
    
    data_params = prepare_data_params(df_long)
    fvt1 = compute_future_values(flexible_logit_model, data_params)
    
    println("Debug: Length of fvt1: ", length(fvt1))
    println("Debug: Number of rows in df_long: ", size(df_long, 1))
    println("Debug: Column names in df_long: ", names(df_long))
    
    structural_params = estimate_structural_parameters(df_long, fvt1)
    
    return structural_params
end
# Wrapper function
function run_ccp_estimation()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df_long = load_and_reshape_data(url)
    
    flexible_logit_model = estimate_flexible_logit(df_long)
    
    data_params = prepare_data_params(df_long)
    fvt1 = compute_future_values(flexible_logit_model, data_params)
    
    println("Debug: Length of fvt1: ", length(fvt1))
    println("Debug: Number of rows in df_long: ", size(df_long, 1))
    
    structural_params = estimate_structural_parameters(df_long, fvt1)
    
    return structural_params
end
# Run the estimation
@time results = run_ccp_estimation()
println(results)

# Unit tests
@testset "CCP Estimation Tests" begin
    @testset "load_and_reshape_data" begin
        df_long = load_and_reshape_data("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv")
        @test size(df_long, 2) == 6  # Assuming 6 columns: bus_id, time, Y, Odometer, RouteUsage, Branded
        @test :Y in names(df_long)
        @test :Odometer in names(df_long)
    end

    @testset "estimate_flexible_logit" begin
        test_df_long = DataFrame(Y=[0,1,1,0], Odometer=[100,200,300,400], RouteUsage=[1,2,1,2], Branded=[0,1,0,1], time=[1,2,3,4])
        model = estimate_flexible_logit(test_df_long)
        @test isa(model, GLM.GeneralizedLinearModel)
    end

    @testset "prepare_data_params" begin
        test_df_long = DataFrame(bus_id=repeat(1:2, inner=2), time=repeat(1:2, outer=2),
                                 Y=[0,1,1,0], Odometer=[100,200,300,400], 
                                 RouteUsage=[1,1,2,2], Branded=[0,0,1,1])
        params = prepare_data_params(test_df_long)
        @test params.N == 2
        @test params.T == 2
    end

    @testset "compute_future_values" begin
        test_df_long = DataFrame(bus_id=repeat(1:2, inner=2), time=repeat(1:2, outer=2),
                                 Y=[0,1,1,0], Odometer=[100,200,300,400], 
                                 RouteUsage=[1,1,2,2], Branded=[0,0,1,1])
        flex_model = estimate_flexible_logit(test_df_long)
        params = prepare_data_params(test_df_long)
        fv = compute_future_values(flex_model, params)
        @test length(fv) == params.N * params.T
    end

    @testset "estimate_structural_parameters" begin
        test_df_long = DataFrame(Y=[0,1,1,0], Odometer=[100,200,300,400], Branded=[0,1,0,1])
        test_fvt1 = [0.1, 0.2, 0.3, 0.4]
        model = estimate_structural_parameters(test_df_long, test_fvt1)
        @test isa(model, GLM.GeneralizedLinearModel)
    end
end