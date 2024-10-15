using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Statistics

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare the data
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = Float64.(df.married.==1)  # Convert to Float64 for numerical stability

# Define the GMM objective function
function gmm_objective(β, X, y)
    N, K = size(X)
    ε = y - X * β  # Residuals
    g = X .* ε  # Moment conditions
    gbar = mean(g, dims=1)  # Average of moment conditions
    return (gbar * gbar')[1]  # Scalar output
end

# Use Optim to estimate GMM coefficients
β_initial = zeros(size(X, 2))
result_gmm = optimize(β -> gmm_objective(β, X, y), β_initial, LBFGS())

# Print GMM estimates
println("GMM estimates using Optim:")
println(Optim.minimizer(result_gmm))

# Check results using closed-form OLS formula
β_ols = inv(X'X) * X'y
println("\nOLS estimates using closed-form formula:")
println(β_ols)

# Compare results
println("\nDifference between GMM and OLS estimates:")
println(Optim.minimizer(result_gmm) - β_ols)
####Question 2#####
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, Statistics, FreqTables

# Load and prepare data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Clean and prepare data
df = dropmissing(df, :occupation)
df[df.occupation .∈ Ref(8:13), :occupation] .= 7

# Manually create categories
unique_occupations = sort(unique(df.occupation))
occupation_dict = Dict(value => index for (index, value) in enumerate(unique_occupations))
df.occupation_code = [occupation_dict[val] for val in df.occupation]

println("Levels of occupation: ", unique_occupations)
println("Number of categories: ", length(unique_occupations))

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation_code
J = length(unique_occupations)

# Helper functions
function mnl_probs(β, X, J)
    K = size(X, 2)
    Z = X * reshape(β, K, J-1)
    Z = hcat(zeros(size(Z, 1)), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    return expZ ./ sum(expZ, dims=2)
end

function mnl_loglikelihood(β, X, y, J)
    P = mnl_probs(β, X, J)
    return -sum(log.(P[i, y[i]]) for i in 1:size(X, 1))
end

function gmm_objective(β, X, y, J)
    N, K = size(X)
    P = mnl_probs(β, X, J)
    d = [y .== j for j in 1:J]
    g = vcat([(d[j] - P[:, j]) .* X for j in 1:J]...)
    gbar = mean(g, dims=1)
    return (gbar * gbar')[1]
end

# (a) Maximum Likelihood Estimation
function mle_multinomial_logit(X, y, J)
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(β -> mnl_loglikelihood(β, X, y, J), β_initial, LBFGS())
    return Optim.minimizer(result)
end

# (b) GMM with MLE starting values
function gmm_mle_start(X, y, J)
    β_mle = mle_multinomial_logit(X, y, J)
    result = optimize(β -> gmm_objective(β, X, y, J), β_mle, LBFGS())
    return Optim.minimizer(result)
end

# (c) GMM with random starting values
function gmm_random_start(X, y, J)
    K = size(X, 2)
    β_random = randn(K * (J - 1))
    result = optimize(β -> gmm_objective(β, X, y, J), β_random, LBFGS())
    return Optim.minimizer(result)
end

# Run estimations
β_mle = mle_multinomial_logit(X, y, J)
β_gmm_mle = gmm_mle_start(X, y, J)
β_gmm_random = gmm_random_start(X, y, J)

# Print results
println("MLE Estimates:")
println(β_mle)
println("\nGMM Estimates (MLE start):")
println(β_gmm_mle)
println("\nGMM Estimates (Random start):")
println(β_gmm_random)

# Compare estimates
println("\nDifference between GMM (MLE start) and GMM (Random start):")
println(β_gmm_mle - β_gmm_random)

# Check if objective function is globally concave
function check_concavity(X, y, J)
    K = size(X, 2)
    trials = 100
    results = zeros(trials)
    for i in 1:trials
        β_random = randn(K * (J - 1))
        result = optimize(β -> gmm_objective(β, X, y, J), β_random, LBFGS())
        results[i] = Optim.minimum(result)
    end
    return all(isapprox.(results, results[1], rtol=1e-5))
end

is_concave = check_concavity(X, y, J)
println("\nIs the objective function globally concave? ", is_concave)
####Question 3####
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, Statistics, Distributions

# Set random seed for reproducibility
Random.seed!(123)

# Function to simulate multinomial logit data
function simulate_mnl_data(N::Int, J::Int, K::Int, β::Vector{Float64})
    # Generate X
    X = [ones(N) randn(N, K-1)]
    
    # Reshape β into a matrix
    β_matrix = reshape(β, K, J-1)
    
    # Calculate probabilities
    Z = X * β_matrix
    Z = hcat(zeros(N), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    P = expZ ./ sum(expZ, dims=2)
    
    # Generate choices
    Y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
    
    return X, Y
end

# Function to estimate multinomial logit model
function estimate_mnl(X::Matrix{Float64}, Y::Vector{Int}, J::Int)
    function mnl_loglikelihood(β)
        K = size(X, 2)
        β_matrix = reshape(β, K, J-1)
        Z = X * β_matrix
        Z = hcat(zeros(size(Z, 1)), Z)  # Add base category
        expZ = exp.(Z .- maximum(Z, dims=2))
        P = expZ ./ sum(expZ, dims=2)
        return -sum(log.(P[i, Y[i]]) for i in 1:size(X, 1))
    end

    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(mnl_loglikelihood, β_initial, LBFGS())
    return Optim.minimizer(result)
end

# Set parameters for simulation
N = 10000  # Number of observations
J = 4      # Number of choice alternatives
K = 3      # Number of covariates (including constant)

# Set true parameter values
β_true = [1.0, -0.5, 0.8, 0.3, 0.6, -0.2, 1.2, -0.7, 0.4]

# Simulate data
X, Y = simulate_mnl_data(N, J, K, β_true)

# Estimate parameters
β_est = estimate_mnl(X, Y, J)

# Compare true and estimated parameters
println("True parameters:")
println(β_true)
println("\nEstimated parameters:")
println(β_est)
println("\nDifference:")
println(β_true - β_est)

# Calculate mean absolute error
mae = mean(abs.(β_true - β_est))
println("\nMean Absolute Error: ", mae)

# Function to calculate choice probabilities
function calculate_probabilities(X::Matrix{Float64}, β::Vector{Float64}, J::Int)
    K = size(X, 2)
    β_matrix = reshape(β, K, J-1)
    Z = X * β_matrix
    Z = hcat(zeros(size(Z, 1)), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    return expZ ./ sum(expZ, dims=2)
end

# Calculate and compare average choice probabilities
P_true = calculate_probabilities(X, β_true, J)
P_est = calculate_probabilities(X, β_est, J)

println("\nAverage True Probabilities:")
println(mean(P_true, dims=1))
println("\nAverage Estimated Probabilities:")
println(mean(P_est, dims=1))
####ًQuestion 5####
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, Statistics, Distributions

# Load and prepare data (same as in Question 2)
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

df = dropmissing(df, :occupation)
df[df.occupation .∈ Ref(8:13), :occupation] .= 7

# Manually create categories
unique_occupations = sort(unique(df.occupation))
occupation_dict = Dict(value => index for (index, value) in enumerate(unique_occupations))
df.occupation_code = [occupation_dict[val] for val in df.occupation]

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation_code
J = length(unique_occupations)
N, K = size(X)

# Function to calculate multinomial logit probabilities
function mnl_probs(β, X, J)
    Z = X * reshape(β, K, J-1)
    Z = hcat(zeros(size(Z, 1)), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    return expZ ./ sum(expZ, dims=2)
end

# SMM objective function
function mnl_smm(θ, X, y, J, S)
    N, K = size(X)
    β = θ
    
    # Data moments: proportion of each choice
    gdata = [mean(y .== j) for j in 1:J]
    
    # Simulated model moments
    gmodel = zeros(J, S)
    Random.seed!(1234)  # Set seed for reproducibility
    
    for s in 1:S
        P = mnl_probs(β, X, J)
        y_sim = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
        gmodel[:, s] = [mean(y_sim .== j) for j in 1:J]
    end
    
    # Criterion function
    err = gdata - mean(gmodel, dims=2)
    J_criterion = err' * err
    return J_criterion[1]  # Return scalar
end

# Estimate using SMM
function estimate_mnl_smm(X, y, J, S)
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(β -> mnl_smm(β, X, y, J, S), β_initial, LBFGS())
    return Optim.minimizer(result)
end

# Run SMM estimation
S = 100  # Number of simulations
β_smm = estimate_mnl_smm(X, y, J, S)

println("SMM Estimates:")
println(β_smm)

# Compare with MLE estimates (from Question 2)
function mle_multinomial_logit(X, y, J)
    function mnl_loglikelihood(β)
        P = mnl_probs(β, X, J)
        return -sum(log(P[i, y[i]]) for i in 1:size(X, 1))
    end
    
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(mnl_loglikelihood, β_initial, LBFGS())
    return Optim.minimizer(result)
end

β_mle = mle_multinomial_logit(X, y, J)

println("\nMLE Estimates:")
println(β_mle)

println("\nDifference between SMM and MLE estimates:")
println(β_smm - β_mle)

# Calculate and compare choice probabilities
P_smm = mnl_probs(β_smm, X, J)
P_mle = mnl_probs(β_mle, X, J)

println("\nAverage SMM Probabilities:")
println(mean(P_smm, dims=1))
println("\nAverage MLE Probabilities:")
println(mean(P_mle, dims=1))
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, Statistics, Distributions, Test

# Main function wrapping all code (Question 6)
function run_problem_set_7()
    # Load and prepare data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    df = dropmissing(df, :occupation)
    df[df.occupation .∈ Ref(8:13), :occupation] .= 7
    
    unique_occupations = sort(unique(df.occupation))
    occupation_dict = Dict(value => index for (index, value) in enumerate(unique_occupations))
    df.occupation_code = [occupation_dict[val] for val in df.occupation]
    
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation_code
    J = length(unique_occupations)
    
    # Question 1: GMM estimation
    β_gmm = gmm_linear_regression(X, y)
    println("GMM Linear Regression Estimates:")
    println(β_gmm)
    
    # Question 2: Multinomial Logit Estimation
    β_mle = mle_multinomial_logit(X, y, J)
    β_gmm_mle = gmm_mle_start(X, y, J)
    β_gmm_random = gmm_random_start(X, y, J)
    
    println("\nMultinomial Logit Estimates:")
    println("MLE: ", β_mle)
    println("GMM (MLE start): ", β_gmm_mle)
    println("GMM (Random start): ", β_gmm_random)
    
    # Question 3: Simulate and Estimate Multinomial Logit
    N_sim, J_sim, K_sim = 10000, 4, 3
    β_true = [1.0, -0.5, 0.8, 0.3, 0.6, -0.2, 1.2, -0.7, 0.4]
    X_sim, Y_sim = simulate_mnl_data(N_sim, J_sim, K_sim, β_true)
    β_est = estimate_mnl(X_sim, Y_sim, J_sim)
    
    println("\nSimulated Multinomial Logit:")
    println("True parameters: ", β_true)
    println("Estimated parameters: ", β_est)
    
    # Question 5: SMM Estimation
    S = 100  # Number of simulations
    β_smm = estimate_mnl_smm(X, y, J, S)
    
    println("\nSMM Estimates:")
    println(β_smm)
end

# Helper functions

function gmm_linear_regression(X, y)
    function gmm_objective(β, X, y)
        ε = y - X * β
        g = X' * ε
        return (g' * g)[1]
    end
    β_initial = zeros(size(X, 2))
    result = optimize(β -> gmm_objective(β, X, y), β_initial, LBFGS())
    return Optim.minimizer(result)
end

function mnl_probs(β, X, J)
    K = size(X, 2)
    Z = X * reshape(β, K, J-1)
    Z = hcat(zeros(size(Z, 1)), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    return expZ ./ sum(expZ, dims=2)
end

function mle_multinomial_logit(X, y, J)
    function mnl_loglikelihood(β)
        P = mnl_probs(β, X, J)
        return -sum(log(P[i, y[i]]) for i in 1:size(X, 1))
    end
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(mnl_loglikelihood, β_initial, LBFGS())
    return Optim.minimizer(result)
end

function gmm_mle_start(X, y, J)
    β_mle = mle_multinomial_logit(X, y, J)
    function gmm_objective(β)
        P = mnl_probs(β, X, J)
        d = [y .== j for j in 1:J]
        g = vcat([(d[j] - P[:, j]) .* X for j in 1:J]...)
        gbar = mean(g, dims=1)
        return (gbar * gbar')[1]
    end
    result = optimize(gmm_objective, β_mle, LBFGS())
    return Optim.minimizer(result)
end

function gmm_random_start(X, y, J)
    K = size(X, 2)
    β_random = randn(K * (J - 1))
    function gmm_objective(β)
        P = mnl_probs(β, X, J)
        d = [y .== j for j in 1:J]
        g = vcat([(d[j] - P[:, j]) .* X for j in 1:J]...)
        gbar = mean(g, dims=1)
        return (gbar * gbar')[1]
    end
    result = optimize(gmm_objective, β_random, LBFGS())
    return Optim.minimizer(result)
end

function simulate_mnl_data(N::Int, J::Int, K::Int, β::Vector{Float64})
    X = [ones(N) randn(N, K-1)]
    β_matrix = reshape(β, K, J-1)
    Z = X * β_matrix
    Z = hcat(zeros(N), Z)  # Add base category
    expZ = exp.(Z .- maximum(Z, dims=2))
    P = expZ ./ sum(expZ, dims=2)
    Y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
    return X, Y
end

function estimate_mnl(X::Matrix{Float64}, Y::Vector{Int}, J::Int)
    function mnl_loglikelihood(β)
        P = mnl_probs(β, X, J)
        return -sum(log(P[i, Y[i]]) for i in 1:size(X, 1))
    end
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(mnl_loglikelihood, β_initial, LBFGS())
    return Optim.minimizer(result)
end

function estimate_mnl_smm(X, y, J, S)
    function mnl_smm(θ)
        β = θ
        gdata = [mean(y .== j) for j in 1:J]
        gmodel = zeros(J, S)
        Random.seed!(1234)
        for s in 1:S
            P = mnl_probs(β, X, J)
            y_sim = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:size(X, 1)]
            gmodel[:, s] = [mean(y_sim .== j) for j in 1:J]
        end
        err = gdata - mean(gmodel, dims=2)
        return (err' * err)[1]
    end
    K = size(X, 2)
    β_initial = zeros(K * (J - 1))
    result = optimize(mnl_smm, β_initial, LBFGS())
    return Optim.minimizer(result)
end

# Unit tests (Question 7)
@testset "Problem Set 7 Tests" begin
    @testset "GMM Linear Regression" begin
        X = [ones(100) randn(100)]
        y = 2 .+ 3 .* X[:, 2] .+ randn(100)
        β_gmm = gmm_linear_regression(X, y)
        @test length(β_gmm) == 2
        @test isapprox(β_gmm[1], 2, atol=0.5)
        @test isapprox(β_gmm[2], 3, atol=0.5)
    end

    @testset "Multinomial Logit Probabilities" begin
        X = [ones(10) randn(10)]
        β = [0.5, -0.3, 0.2]
        J = 3
        P = mnl_probs(β, X, J)
        @test size(P) == (10, 3)
        @test all(sum(P, dims=2) .≈ 1)
    end

    @testset "Simulate Multinomial Logit Data" begin
        N, J, K = 1000, 4, 3
        β = rand(K * (J - 1))
        X, Y = simulate_mnl_data(N, J, K, β)
        @test size(X) == (N, K)
        @test length(Y) == N
        @test all(1 .<= Y .<= J)
    end

    @testset "Estimate Multinomial Logit" begin
        N, J, K = 1000, 3, 2
        β_true = [0.5, -0.3, 0.2, 0.1]
        X, Y = simulate_mnl_data(N, J, K, β_true)
        β_est = estimate_mnl(X, Y, J)
        @test length(β_est) == length(β_true)
        @test isapprox(β_est, β_true, atol=0.5)
    end

    @testset "SMM Estimation" begin
        N, J, K = 500, 3, 2
        β_true = [0.5, -0.3, 0.2, 0.1]
        X, Y = simulate_mnl_data(N, J, K, β_true)
        β_smm = estimate_mnl_smm(X, Y, J, 50)
        @test length(β_smm) == length(β_true)
        @test isapprox(β_smm, β_true, atol=0.5)
    end
end

# Run the main function
run_problem_set_7()

# Run the tests
runtests()