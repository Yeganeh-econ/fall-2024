using Pkg
Pkg.add("Optim")
using Optim
# Define the function to maximize
f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2

# Define the negative of the function for minimization
minusf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
# Starting value
startval = rand(1) # generates a random starting value
# Perform the optimization
result = optimize(minusf, startval, BFGS())
# Print the result
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))
# part 2
Pkg.add(["Optim", "DataFrames", "CSV", "HTTP", "GLM"])
using Optim, DataFrames, CSV, HTTP, GLM
# Import and load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
# Create matrix of regressors X and dependent variable y
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1
# Define the OLS objective function (Sum of Squared Residuals)
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end
# Perform the optimization
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

# Print the estimated coefficients
println("Estimated OLS coefficients: ", beta_hat_ols.minimizer)
# OLS estimates using matrix algebra
bols = inv(X' * X) * X' * y
println("OLS estimates using matrix algebra: ", bols)

# OLS estimates using the GLM package
df.white = df.race .== 1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("OLS estimates using GLM: ", coef(bols_lm))
using Optim, DataFrames, CSV, HTTP, GLM

# Import and load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Create matrix of regressors X and dependent variable y
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1

# Define the OLS objective function (Sum of Squared Residuals)
function ols(beta, X, y)
    ssr = sum((y .- X * beta).^2)
    return ssr
end

# Perform the optimization
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

# Print the estimated coefficients
println("Estimated OLS coefficients: ", beta_hat_ols.minimizer)

# OLS estimates using matrix algebra
bols = inv(X' * X) * X' * y
println("OLS estimates using matrix algebra: ", bols)

# OLS estimates using the GLM package
df.white = df.race .== 1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("OLS estimates using GLM: ", coef(bols_lm))
using Optim, DataFrames, CSV, HTTP, GLM, LinearAlgebra
# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Prepare matrix X and dependent variable y
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.married .== 1
# Define the logistic function (logit model)
function logit_log_likelihood(beta, X, y)
    p = 1 ./ (1 .+ exp.(-X * beta))  # logistic function
    ll = sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))  # log-likelihood
    return -ll  # return negative log-likelihood for minimization
end
# Perform the optimization
beta_hat_logit = optimize(b -> logit_log_likelihood(b, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))

# Print the estimated coefficients
println("Estimated Logit coefficients: ", beta_hat_logit.minimizer)
# Logit estimates using the GLM package
df.white = df.race .== 1
logit_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println("Logit estimates using GLM: ", coef(logit_glm))
using Optim, DataFrames, CSV, HTTP, GLM, LinearAlgebra, FreqTables
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, d)

    # your turn

    return loglike
end

# Load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Clean the occupation variable by dropping missing values
df = dropmissing(df, :occupation)

# Aggregate occupation categories
df[df.occupation .== 8, :occupation] .= 7
df[df.occupation .== 9, :occupation] .= 7
df[df.occupation .== 10, :occupation] .= 7
df[df.occupation .== 11, :occupation] .= 7
df[df.occupation .== 12, :occupation] .= 7
df[df.occupation .== 13, :occupation] .= 7

# Check the frequency of occupation categories
freqtable(df, :occupation)

# Prepare matrix of regressors X and dependent variable y
X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
y = df.occupation

# Define the negative log-likelihood for multinomial logit
function mlogit_neg_log_likelihood(beta, X, y, J)
    N, K = size(X)  # Number of observations and covariates
    # Reshape beta into K × (J-1) matrix
    beta_matrix = reshape(beta, K, J - 1)

    # Debugging: Check the size of reshaped beta_matrix
    println("Reshaped beta_matrix size: ", size(beta_matrix))

    ll = 0.0  # Log-likelihood accumulator
    
    for i in 1:N
        # Calculate probabilities for each choice alternative
        num = exp.(X[i, :] * beta_matrix)  # Numerators for categories 1 to J-1
        
        # Debugging: Check the dimensions during multiplication
        println("Dimensions of X[i, :]: ", size(X[i, :]))
        println("Dimensions of num (probabilities): ", size(num))
        
        denom = 1 + sum(num)  # Denominator includes the base category
        probs = vcat(1 / denom, num / denom)  # Probabilities for each category
        
        # Debugging: Check the dimension of probs
        println("Dimensions of probs: ", size(probs))

        # Contribution to the log-likelihood
        ll += log(probs[y[i]])
    end
    
    return -ll  # Return the negative log-likelihood for minimization
end

# Number of categories (J)
J = length(unique(y))

# Perform the optimization
initial_beta = rand(size(X, 2) * (J - 1))  # Starting values
result = optimize(b -> mlogit_neg_log_likelihood(b, X, y, J), initial_beta, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))

# Print the estimated coefficients
println("Estimated Multinomial Logit coefficients: ", result.minimizer)
using Optim, DataFrames, CSV, HTTP, LinearAlgebra, FreqTables

# Function to load the data, clean it, and perform the multinomial logit estimation
function estimate_multinomial_logit()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Clean the occupation variable by dropping missing values
    df = dropmissing(df, :occupation)

    # Aggregate occupation categories
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    # Prepare matrix of regressors X and dependent variable y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    # Define the negative log-likelihood for multinomial logit
    function mlogit_neg_log_likelihood(beta, X, y, J)
        N, K = size(X)  # Number of observations and covariates
        beta_matrix = reshape(beta, K, J - 1)  # Reshape beta into K × (J-1) matrix
        ll = 0.0  # Log-likelihood accumulator
        
        for i in 1:N
            num = exp.(X[i, :] * beta_matrix)  # Numerators for categories 1 to J-1
            denom = 1 + sum(num)  # Denominator includes the base category
            probs = vcat(1 / denom, num / denom)  # Probabilities for each category
            ll += log(probs[y[i]])
        end
        
        return -ll  # Return the negative log-likelihood for minimization
    end

    # Number of categories (J)
    J = length(unique(y))

    # Perform the optimization
    initial_beta = rand(size(X, 2) * (J - 1))  # Starting values
    result = optimize(b -> mlogit_neg_log_likelihood(b, X, y, J), initial_beta, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))

    # Return the estimated coefficients
    return result.minimizer
end

# Call the function and print the results
println("Estimated Multinomial Logit coefficients: ", estimate_multinomial_logit())
using Optim, DataFrames, CSV, HTTP, LinearAlgebra, FreqTables, Test

# Function to load the data, clean it, and perform the multinomial logit estimation
function estimate_multinomial_logit()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # Clean the occupation variable by dropping missing values
    df = dropmissing(df, :occupation)

    # Aggregate occupation categories
    df[df.occupation .== 8, :occupation] .= 7
    df[df.occupation .== 9, :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    # Prepare matrix of regressors X and dependent variable y
    X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    # Define the negative log-likelihood for multinomial logit
    function mlogit_neg_log_likelihood(beta, X, y, J)
        N, K = size(X)  # Number of observations and covariates
        beta_matrix = reshape(beta, K, J - 1)  # Reshape beta into K × (J-1) matrix
        ll = 0.0  # Log-likelihood accumulator
        
        for i in 1:N
            num = exp.(X[i, :] * beta_matrix)  # Numerators for categories 1 to J-1
            denom = 1 + sum(num)  # Denominator includes the base category
            probs = vcat(1 / denom, num / denom)  # Probabilities for each category
            ll += log(probs[y[i]])
        end
        
        return -ll  # Return the negative log-likelihood for minimization
    end

    # Number of categories (J)
    J = length(unique(y))

    # Perform the optimization
    initial_beta = rand(size(X, 2) * (J - 1))  # Starting values
    result = optimize(b -> mlogit_neg_log_likelihood(b, X, y, J), initial_beta, LBFGS(), Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))

    # Return the estimated coefficients
    return result.minimizer
end

# Unit tests

@testset "Multinomial Logit Estimation Tests" begin

    # Test if the function loads the data correctly
    @testset "Data Loading" begin
        df = CSV.read(HTTP.get("https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv").body, DataFrame)
        @test !ismissing(df)  # Test if data loaded without missing data
        @test size(df, 1) > 0  # Test that the dataset has rows
        @test size(df, 2) > 0  # Test that the dataset has columns
    end

    # Test if the occupation cleaning step works correctly
    @testset "Occupation Cleaning" begin
        df = dropmissing(df, :occupation)
        df[df.occupation .== 8, :occupation] .= 7
        df[df.occupation .== 9, :occupation] .= 7
        df[df.occupation .== 10, :occupation] .= 7
        df[df.occupation .== 11, :occupation] .= 7
        df[df.occupation .== 12, :occupation] .= 7
        df[df.occupation .== 13, :occupation] .= 7
        @test all(!ismissing(df.occupation))  # Test that all missing occupations were removed
        @test length(unique(df.occupation)) <= 7  # Test that there are no more than 7 categories after aggregation
    end

    # Test if the regression design matrix X and the response y are correctly formed
    @testset "Matrix Preparation" begin
        X = [ones(size(df, 1)) df.age df.race .== 1 df.collgrad .== 1]
        y = df.occupation
        @test size(X, 1) == size(df, 1)  # Test that the number of rows in X matches the number of observations
        @test size(X, 2) == 4  # Test that X has 4 columns (including the constant, age, race, and collgrad)
        @test size(y, 1) == size(df, 1)  # Test that the dependent variable y has the correct number of rows
    end

    # Test if the mlogit_neg_log_likelihood function behaves as expected
    @testset "Log Likelihood Calculation" begin
        X = [ones(100) randn(100, 3)]  # Mock data for testing
        y = rand(1:4, 100)  # Random categories 1 to 4
        beta = rand(4 * 3)  # Random beta values
        ll_value = mlogit_neg_log_likelihood(beta, X, y, 4)
        @test ll_value < 0  # Log-likelihood should be negative
    end

    # Test the optimization result
    @testset "Optimization" begin
        # Check if the function returns valid estimated coefficients
        estimated_coeffs = estimate_multinomial_logit()
        @test length(estimated_coeffs) == 12  # Should match K * (J-1)
        @test all(isfinite, estimated_coeffs)  # Ensure that the coefficients are finite
    end
end

