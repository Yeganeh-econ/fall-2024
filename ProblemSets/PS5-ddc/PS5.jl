using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM
import Pkg
Pkg.add("DataFramesMeta")
using DataFramesMeta: @transform, @select
# Include the create_grids function
include("create_grids.jl")
using .Main: create_grids

function main()
    # Question 1: reshaping the data
    url_static = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url_static).body, DataFrame)

    # Add bus_id column
    df.bus_id = 1:nrow(df)
    df = @transform(df,:bus_id = 1:size(df,1))
    # Define y_cols and odo_cols
    y_cols = [Symbol("Y$i") for i in 1:20]
    odo_cols = [Symbol("Odo$i") for i in 1:20]

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one 
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long,:time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))
    
    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))
    
    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])
    # Question 2: estimate a static version of the model
    static_model = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink())
    println("Static Model Results:")
    println(static_model)

    # Question 3a: read in data for dynamic model
    url_dynamic = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
    df_dynamic = CSV.read(HTTP.get(url_dynamic).body, DataFrame)

    Y = Matrix(df_dynamic[:, y_cols])
    Odo = Matrix(df_dynamic[:, odo_cols])
    Xst = Matrix(df_dynamic[:, [Symbol("Xst$i") for i in 1:20]])

    # Question 3b: generate state transition matrices
    zval,zbin,xval,xbin,xtran = create_grids()

    # Estimate the dynamic model
    initial_θ = coef(static_model)
    β = 0.9
    T = 20

    # Print diagnostic information
    println("Initial θ: ", initial_θ)
    println("Size of Y: ", size(Y))
    println("Size of Odo: ", size(Odo))
    println("Size of Xst: ", size(Xst))
    println("Range of Odo: ", extrema(Odo))
    println("Range of Xst: ", extrema(Xst))

    result = estimate_dynamic_model(df_dynamic, Y, Odo, Xst, β, T, initial_θ)
    println("\nDynamic Model Results:")
    println(result)

    test_compute_future_values()
end

# Function to compute future values
function compute_future_values(θ, β, T, zbin, xbin, xval, xtran)
    FV = zeros(zbin*xbin, 2, T+1)
    
    for t in T:-1:1
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    row = x + (z-1)*xbin
                    
                    # Compute v1t
                    v1t = θ[1] + θ[2]*xval[x] + θ[3]*b + 
                          β * (xtran[row,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                    
                    # Compute v0t
                    v0t = β * (xtran[1+(z-1)*xbin,:]' * FV[(z-1)*xbin+1:z*xbin, b+1, t+1])
                    
                    # Update future value
                    FV[row, b+1, t] = β * log(exp(v0t) + exp(v1t))
                end
            end
        end
    end
    
    return FV
end

# Function to compute log-likelihood
function log_likelihood(θ, df, Y, Odo, Xst, β, T, zbin, xbin, xval, xtran, FV)
    ll = 0.0
    N = size(df, 1)
    
    for i in 1:N
        for t in 1:T
            row0 = 1 + Int(round(Xst[i,t])) - 1
            row1 = Int(round(Odo[i,t])) + (Int(round(Xst[i,t])) - 1) * xbin
            
            row0 = max(1, min(row0, size(FV, 1)))
            row1 = max(1, min(row1, size(xtran, 1)))
            
            v1t_v0t = θ[1] + θ[2]*Odo[i,t] + θ[3]*df.Branded[i] + 
                      β * ((xtran[row1,:].-xtran[row0,:])' * FV[row0:min(row0+xbin-1, size(FV, 1)), df.Branded[i]+1, t+1])
            
            p1t = exp(v1t_v0t) / (1 + exp(v1t_v0t))
            
            ll += Y[i,t] * log(p1t) + (1 - Y[i,t]) * log(1 - p1t)
        end
    end
    
    return -ll  # Return negative log-likelihood for minimization
end

# Main estimation function
@views @inbounds function estimate_dynamic_model(df, Y, Odo, Xst, β, T, initial_θ)
    zval, zbin, xval, xbin, xtran = create_grids()
    
    function objective(θ)
        try
            FV = compute_future_values(θ, β, T, zbin, xbin, xval, xtran)
            return log_likelihood(θ, df, Y, Odo, Xst, β, T, zbin, xbin, xval, xtran, FV)
        catch e
            println("Error in objective function: ", e)
            return Inf  # Return a large value if there's an error
        end
    end
    
    result = optimize(objective, initial_θ, BFGS())
    
    return result
end

# Unit tests
function test_compute_future_values()
    # Set up test data
    θ_test = [1.0, -0.1, 0.5]
    β_test = 0.9
    T_test = 5
    zbin_test = 3
    xbin_test = 4
    xval_test = [0.0, 0.125, 0.25, 0.375]
    xtran_test = rand(zbin_test * xbin_test, xbin_test)
    xtran_test = xtran_test ./ sum(xtran_test, dims=2)  # Normalize rows to sum to 1
    
    # Run the function
    FV_test = compute_future_values(θ_test, β_test, T_test, zbin_test, xbin_test, xval_test, xtran_test)
    
    # Check dimensions
    @assert size(FV_test) == (zbin_test * xbin_test, 2, T_test + 1) "Incorrect dimensions of FV"
    
    # Check if values are reasonable (between -100 and 100, for example)
    @assert all(-100 .<= FV_test .<= 100) "FV values out of expected range"
    
    println("All tests passed for compute_future_values!")
end

# Call the main function
main()