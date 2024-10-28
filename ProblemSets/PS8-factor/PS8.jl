# Install and load required packages
using Pkg
using Distributed  # Added for parallelization

# List of packages to install
packages = [
    "Random",
    "LinearAlgebra",
    "Statistics",
    "Optim",
    "DataFrames",
    "CSV",
    "HTTP",
    "GLM",
    "MultivariateStats",
    "Test",
    "Distributions"
]

# Install packages if not already installed
println("Checking and installing required packages...")
for package in packages
    try
        @eval using $(Symbol(package))
    catch
        println("Installing $package...")
        Pkg.add(package)
    end
end

# Load all packages
using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using MultivariateStats
using Test
using Distributions

# Helper function to load data
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    return df
end

# Question 1: Basic linear regression
function question1(df)
    println("\nQuestion 1: Basic Linear Regression")
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
    println(model)
    return model
end

# Question 2: Compute correlation among ASVAB variables
function question2(df)
    println("\nQuestion 2: ASVAB Correlation Matrix")
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    cor_matrix = cor(Matrix(df[:, asvab_vars]))
    println("\nCorrelation matrix of ASVAB variables:")
    display(cor_matrix)
    return cor_matrix
end

# Question 3: Regression with ASVAB variables
function question3(df)
    println("\nQuestion 3: Regression with ASVAB variables")
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                       asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
    println(model)
    return model
end

# Question 4: PCA regression
function question4(df)
    println("\nQuestion 4: PCA Regression")
    
    # Extract ASVAB variables
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvab_mat = Matrix(df[:, asvab_vars])'  # Transpose for MultivariateStats
    
    # Fit PCA
    M = fit(PCA, asvab_mat; maxoutdim=1)
    
    # Get first principal component
    asvab_pca = vec(MultivariateStats.transform(M, asvab_mat)')
    
    # Add PCA scores to dataframe
    df_pca = copy(df)
    df_pca.asvab_pc1 = asvab_pca
    
    # Run regression with PCA
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_pc1), df_pca)
    println(model)
    
    # Print PCA details
    println("\nPCA Details:")
    println("Explained variance ratio: ", principalratio(M))
    println("Principal axes:")
    display(projection(M))
    
    return model, M
end

# Question 5: Factor Analysis regression
function question5(df)
    println("\nQuestion 5: Factor Analysis Regression")
    
    # Extract ASVAB variables
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    asvab_mat = Matrix(df[:, asvab_vars])'  # Transpose for MultivariateStats
    
    # Fit Factor Analysis
    M = fit(FactorAnalysis, asvab_mat; maxoutdim=1)
    
    # Get factor scores
    asvab_fa = vec(MultivariateStats.transform(M, asvab_mat)')
    
    # Add FA scores to dataframe
    df_fa = copy(df)
    df_fa.asvab_factor = asvab_fa
    
    # Run regression with FA
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvab_factor), df_fa)
    println(model)
    
    # Print FA details
    println("\nFactor Analysis Details:")
    println("Loading matrix:")
    display(M.W)
    
    return model, M
end

# Helper function for Gauss-Legendre quadrature
@everywhere function lgwt(N::Int, a::Float64, b::Float64)
    function roots(n::Int)
        i = 1:n-1
        v = i ./ sqrt.((2 .* i).^2 .- 1)
        T = diagm(-1 => v, 1 => v)
        d, V = eigen(T)
        x = d[1:n]
        w = 2 .* (V[1, 1:n]).^2
        return x, w
    end
    
    x, w = roots(N)
    x = (b-a)/2 .* x .+ (a+b)/2
    w = w .* (b-a)/2
    return x, w
end

# Optimized log likelihood function for full measurement system
@everywhere function log_likelihood(theta, df, R)
    # Extract parameters
    α₀ = theta[1:6]     # Intercepts for ASVAB equations
    α₁ = theta[7:12]    # Black coefficients
    α₂ = theta[13:18]   # Hispanic coefficients
    α₃ = theta[19:24]   # Female coefficients
    γ  = theta[25:30]   # Factor loadings
    β  = theta[31:37]   # Wage equation coefficients
    δ  = theta[38]      # Factor loading in wage equation
    σ_asvab = exp.(theta[39:44])  # ASVAB error standard deviations
    σ_wage = exp(theta[45])       # Wage equation error standard deviation

    # Get quadrature nodes and weights
    nodes, weights = lgwt(R, -4.0, 4.0)

    # Extract variables outside of the main loop
    asvab_vars = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
    X = [ones(nrow(df)) df.black df.hispanic df.female]
    X_wage = [ones(nrow(df)) df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr]

    # Parallelized and vectorized log likelihood
    ll = @distributed (+) for i in 1:nrow(df)
        log_integral_terms = map((node, weight) -> begin
            # ASVAB likelihood
            asvab_ll = sum(logpdf(Normal(dot(X[i, :], [α₀[j], α₁[j], α₂[j], α₃[j]]) + γ[j]*node, σ_asvab[j]), df[i, asvab_vars[j]]) for j in 1:6)

            # Wage likelihood
            wage_ll = logpdf(Normal(dot(X_wage[i, :], β) + δ*node, σ_wage), df.logwage[i])

            # Return the log of the integral term
            log(weight) + asvab_ll + wage_ll + logpdf(Normal(0, 1), node)
        end, nodes, weights)

        max_term = maximum(log_integral_terms)
        max_term + log(sum(exp.(log_integral_terms .- max_term)))
    end

    return -ll
end

# Question 6: Full measurement system MLE
function question6(df)
    println("\nQuestion 6: Full Measurement System MLE")
    
    # Initial parameter values - more careful initialization
    θ₀ = vcat(
        zeros(6),    # α₀: ASVAB intercepts
        zeros(6),    # α₁: Black coefficients
        zeros(6),    # α₂: Hispanic coefficients
        zeros(6),    # α₃: Female coefficients
        ones(6),     # γ: Factor loadings initialized to 1
        [0.0, -0.1, -0.1, -0.1, 0.1, 0.0, 0.0],  # β: Wage equation coefficients
        1.0,         # δ: Wage factor loading
        fill(log(1.0), 6),  # log(σ_asvab): Log ASVAB standard deviations
        log(1.0)     # log(σ_wage): Log wage standard deviation
    )
    
    # Optimization settings
    R = 5  # Reduced number of quadrature points for efficiency
    opt_settings = Optim.Options(
        iterations=1000,
        show_trace=true,
        show_every=50,
        g_tol=1e-4,
        f_tol=1e-6,
        x_tol=1e-4
    )
    
    # Try optimization with multiple algorithms
    algorithms = [
        LBFGS(),
        NelderMead(),
        SimulatedAnnealing()
    ]
    
    best_result = nothing
    best_value = Inf
    
    for algorithm in algorithms
        try
            println("\nTrying optimization with ", typeof(algorithm))
            result = optimize(θ -> log_likelihood(θ, df, R), θ₀, algorithm, opt_settings)
            
            if Optim.minimum(result) < best_value
                best_result = result
                best_value = Optim.minimum(result)
            end
            
            if Optim.converged(result)
                println("Converged with ", typeof(algorithm))
                break
            end
        catch e
            println("Error with ", typeof(algorithm), ": ", e)
            continue
        end
    end
    
    if best_result === nothing
        error("All optimization attempts failed")
    end
    
    # Extract and print results
    θ_hat = Optim.minimizer(best_result)
    
    println("\nOptimization Results:")
    println("Convergence: ", Optim.converged(best_result))
    println("Minimum value: ", Optim.minimum(best_result))
    
    # Print parameter estimates with labels
    param_names = [
        "ASVAB intercepts (α₀)", "Black coefficients (α₁)", 
        "Hispanic coefficients (α₂)", "Female coefficients (α₃)",
        "Factor loadings (γ)", "Wage equation coefficients (β)",
        "Wage factor loading (δ)", "ASVAB std deviations (σ_asvab)",
        "Wage std deviation (σ_wage)"
    ]
    
    println("\nParameter Estimates:")
    idx = 1
    for (i, name) in enumerate(param_names)
        if i == 6  # Wage equation coefficients
            println(name, ":")
            println("  Intercept: ", θ_hat[idx])
            println("  Black: ", θ_hat[idx+1])
            println("  Hispanic: ", θ_hat[idx+2])
            println("  Female: ", θ_hat[idx+3])
            println("  Schooling: ", θ_hat[idx+4])
            println("  High School: ", θ_hat[idx+5])
            println("  College: ", θ_hat[idx+6])
            idx += 7
        elseif i == 7  # Single parameter
            println(name, ": ", θ_hat[idx])
            idx += 1
        elseif i == 8  # Standard deviations need exp
            println(name, ": ", exp.(θ_hat[idx:idx+5]))
            idx += 6
        elseif i == 9  # Wage std deviation needs exp
            println(name, ": ", exp(θ_hat[idx]))
            idx += 1
        else  # Arrays of 6
            println(name, ": ", θ_hat[idx:idx+5])
            idx += 6
        end
    end
    
    return best_result, θ_hat
end

# Main function to run all analyses
function main()
    println("Loading data...")
    df = load_data()
    
    q1_model = question1(df)
    q2_corr = question2(df)
    q3_model = question3(df)
    q4_model, pca_fit = question4(df)
    q5_model, fa_fit = question5(df)
    q6_results, θ_hat = question6(df)
end

# Run the analysis
main()

