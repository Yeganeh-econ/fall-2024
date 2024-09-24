using Optim, DataFrames, CSV, HTTP, GLM, LinearAlgebra, Random, Statistics, FreqTables, Test

function allwrap()
    # Load the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    # Question 1: Multinomial Logit with alternative-specific covariates
    function mnl_loglikelihood(params, X, Z, y)
        n_obs, n_choices = size(Z)
        n_vars = size(X, 2)
        
        β = reshape(params[1:end-1], n_vars, n_choices-1)
        γ = params[end]
        
        loglik = 0.0
        for i in 1:n_obs
            denom = 1.0
            for j in 1:n_choices-1
                denom += exp(dot(X[i,:], β[:,j]) + γ * (Z[i,j] - Z[i,end]))
            end
            
            if y[i] == n_choices
                loglik += -log(denom)
            else
                loglik += dot(X[i,:], β[:,y[i]]) + γ * (Z[i,y[i]] - Z[i,end]) - log(denom)
            end
        end
        
        return -loglik
    end

    # Initialize parameters
    n_vars = size(X, 2)
    n_choices = size(Z, 2)
    initial_params = vcat(vec(zeros(n_vars, n_choices-1)), 0.0)

    # Optimize
    result_mnl = optimize(params -> mnl_loglikelihood(params, X, Z, y), initial_params, BFGS())

    # Extract and print results
    β_hat = reshape(Optim.minimizer(result_mnl)[1:end-1], n_vars, n_choices-1)
    γ_hat = Optim.minimizer(result_mnl)[end]

    println("Multinomial Logit Results:")
    println("β_hat:")
    display(β_hat)
    println("\nγ_hat: ", γ_hat)

    # Question 2: Interpretation of γ̂
    println("\nInterpretation of γ̂:")
    println("The estimated coefficient γ̂ = ", γ_hat, " represents the effect of a one-unit change in the difference between an alternative's Z value and the base alternative's Z value on the log-odds of choosing that alternative over the base alternative, holding all else constant.")

    # Question 3: Nested Logit
    function nested_logit_loglikelihood(params, X, Z, y)
        n_obs, n_choices = size(Z)
        n_vars = size(X, 2)
        
        β_WC = params[1:n_vars]
        β_BC = params[n_vars+1:2*n_vars]
        λ_WC = params[2*n_vars+1]
        λ_BC = params[2*n_vars+2]
        γ = params[end]
        
        loglik = 0.0
        for i in 1:n_obs
            if y[i] in [1, 2, 3]  # White Collar
                num = exp((dot(X[i,:], β_WC) + γ * (Z[i,y[i]] - Z[i,end])) / λ_WC)
                denom_WC = sum(exp.((dot(X[i,:], β_WC) .+ γ .* (Z[i,1:3] .- Z[i,end])) ./ λ_WC))
                denom = 1 + denom_WC^λ_WC + sum(exp.((dot(X[i,:], β_BC) .+ γ .* (Z[i,4:7] .- Z[i,end])) ./ λ_BC))^λ_BC
                loglik += log(num) + (λ_WC - 1) * log(denom_WC) - log(denom)
            elseif y[i] in [4, 5, 6, 7]  # Blue Collar
                num = exp((dot(X[i,:], β_BC) + γ * (Z[i,y[i]] - Z[i,end])) / λ_BC)
                denom_BC = sum(exp.((dot(X[i,:], β_BC) .+ γ .* (Z[i,4:7] .- Z[i,end])) ./ λ_BC))
                denom = 1 + sum(exp.((dot(X[i,:], β_WC) .+ γ .* (Z[i,1:3] .- Z[i,end])) ./ λ_WC))^λ_WC + denom_BC^λ_BC
                loglik += log(num) + (λ_BC - 1) * log(denom_BC) - log(denom)
            else  # Other
                denom = 1 + sum(exp.((dot(X[i,:], β_WC) .+ γ .* (Z[i,1:3] .- Z[i,end])) ./ λ_WC))^λ_WC + 
                        sum(exp.((dot(X[i,:], β_BC) .+ γ .* (Z[i,4:7] .- Z[i,end])) ./ λ_BC))^λ_BC
                loglik += -log(denom)
            end
        end
        
        return -loglik
    end

    # Initialize parameters for nested logit
    initial_params_nested = vcat(zeros(2*n_vars), [1.0, 1.0], 0.0)

    # Optimize
    result_nested = optimize(params -> nested_logit_loglikelihood(params, X, Z, y), initial_params_nested, BFGS())

    # Extract and print results
    β_WC_hat = Optim.minimizer(result_nested)[1:n_vars]
    β_BC_hat = Optim.minimizer(result_nested)[n_vars+1:2*n_vars]
    λ_WC_hat = Optim.minimizer(result_nested)[2*n_vars+1]
    λ_BC_hat = Optim.minimizer(result_nested)[2*n_vars+2]
    γ_hat_nested = Optim.minimizer(result_nested)[end]

    println("\nNested Logit Results:")
    println("β_WC_hat:", β_WC_hat)
    println("β_BC_hat:", β_BC_hat)
    println("λ_WC_hat:", λ_WC_hat)
    println("λ_BC_hat:", λ_BC_hat)
    println("γ_hat:", γ_hat_nested)

    # Question 5: Unit tests
    @testset "Multinomial Logit Tests" begin
        # Test with simple data
        X_test = [1.0 0.0; 0.0 1.0]
        Z_test = [1.0 0.0 0.0; 0.0 1.0 0.0]
        y_test = [1, 2]
        params_test = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Test if function runs without error
        @test_nowarn mnl_loglikelihood(params_test, X_test, Z_test, y_test)
        
        # Test if output is a scalar
        @test isa(mnl_loglikelihood(params_test, X_test, Z_test, y_test), Number)
        
        # Test if output is negative (since we're minimizing negative log-likelihood)
        @test mnl_loglikelihood(params_test, X_test, Z_test, y_test) < 0
    end

    @testset "Nested Logit Tests" begin
        # Test with simple data
        X_test = [1.0 0.0; 0.0 1.0]
        Z_test = [1.0 0.0 0.0; 0.0 1.0 0.0]
        y_test = [1, 2]
        params_test = [0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 0.5]
        
        # Test if function runs without error
        @test_nowarn nested_logit_loglikelihood(params_test, X_test, Z_test, y_test)
        
        # Test if output is a scalar
        @test isa(nested_logit_loglikelihood(params_test, X_test, Z_test, y_test), Number)
        
        # Test if output is negative (since we're minimizing negative log-likelihood)
        @test nested_logit_loglikelihood(params_test, X_test, Z_test, y_test) < 0
    end

    println("\nAll tests completed.")
end

# Call the function to run everything
allwrap()