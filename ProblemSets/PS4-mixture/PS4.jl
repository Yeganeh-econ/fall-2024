using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random, Statistics, GLM, FreqTables, Distributions
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code
    return df, X, Z, y
end

function mlogit_with_Z(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum(bigY.*log.(P))
    
    return loglike
end

function optimize_mlogit(X, Z, y)
    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    return theta_hat_mle_ad, theta_hat_mle_ad_se
end
###Question3 part a
using Distributions
include("lgwt.jl")

# Define distribution
d = Normal(0, 2)  # mean=0, standard deviation=2

# Get quadrature nodes and weights for 7 grid points
nodes, weights = lgwt(7, -5*2, 5*2)  # Using ±5σ as bounds

# Compute the integral of x^2 * f(x) over the density
integral_x2 = sum(weights .* (nodes.^2) .* pdf.(d, nodes))
println("Integral of x^2 * f(x) with 7 points: ", integral_x2)

# Compute with 10 quadrature points
nodes10, weights10 = lgwt(10, -5*2, 5*2)
integral_x2_10 = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
println("Integral of x^2 * f(x) with 10 points: ", integral_x2_10)

# True value (variance of N(0,2) distribution)
true_value = var(d)
println("True value (variance): ", true_value)

# Comment on approximation
println("Approximation error (7 points): ", abs(integral_x2 - true_value))
println("Approximation error (10 points): ", abs(integral_x2_10 - true_value))
####Question 3 part b
using Distributions
include("lgwt.jl")

# Define the N(0,2) distribution
d = Normal(0, 2)

# Function to compute the integral using quadrature
function quadrature_integral(n_points)
    # Use 5σ as the integration bounds
    sigma = std(d)
    nodes, weights = lgwt(n_points, -5*sigma, 5*sigma)
    
    # Compute the integral of x^2 * f(x)
    integral = sum(weights .* (nodes.^2) .* pdf.(d, nodes))
    return integral
end

# Compute with 7 quadrature points
integral_7 = quadrature_integral(7)
println("Integral with 7 quadrature points: ", integral_7)

# Compute with 10 quadrature points
integral_10 = quadrature_integral(10)
println("Integral with 10 quadrature points: ", integral_10)

# True value (variance of N(0,2) distribution)
true_variance = var(d)
println("True variance: ", true_variance)

# Comment on approximation
println("Approximation error (7 points): ", abs(integral_7 - true_variance))
println("Approximation error (10 points): ", abs(integral_10 - true_variance))

# Comment on how well the quadrature approximates the true value
println("\nComment on approximation:")
println("The quadrature method provides a very good approximation of the true variance.")
println("With 7 points, the error is already small, and with 10 points, it's even smaller.")
println("This demonstrates that Gauss-Legendre quadrature is highly efficient for this type of integral,")
println("especially when the integrand is well-behaved (as is the case with the normal distribution).")
### Question3 part c
using Distributions, Random

function monte_carlo_integral(f, a, b, D)
    Random.seed!(123)  # for reproducibility
    x = rand(Uniform(a, b), D)
    return (b - a) * mean(f.(x))
end

# Define the N(0,2) distribution
d = Normal(0, 2)

# Integrate x^2 * f(x)
mc_integral_x2_1m = monte_carlo_integral(x -> x^2 * pdf(d, x), -5*std(d), 5*std(d), 1_000_000)
println("Monte Carlo integral of x^2 * f(x) (1,000,000 draws): ", mc_integral_x2_1m)

# Integrate x * f(x)
mc_integral_x_1m = monte_carlo_integral(x -> x * pdf(d, x), -5*std(d), 5*std(d), 1_000_000)
println("Monte Carlo integral of x * f(x) (1,000,000 draws): ", mc_integral_x_1m)

# Integrate f(x)
mc_integral_1_1m = monte_carlo_integral(x -> pdf(d, x), -5*std(d), 5*std(d), 1_000_000)
println("Monte Carlo integral of f(x) (1,000,000 draws): ", mc_integral_1_1m)

# True values
true_variance = var(d)
true_mean = mean(d)
true_integral = 1.0  # integral of a PDF over its support

println("\nTrue values:")
println("Variance (E[X^2]): ", true_variance)
println("Mean (E[X]): ", true_mean)
println("Integral of PDF: ", true_integral)

println("\nApproximation errors:")
println("Variance error: ", abs(mc_integral_x2_1m - true_variance))
println("Mean error: ", abs(mc_integral_x_1m - true_mean))
println("PDF integral error: ", abs(mc_integral_1_1m - true_integral)) 
### Question3 part D
# Compute with 1,000 draws for comparison
mc_integral_x2_1k = monte_carlo_integral(x -> x^2 * pdf(d, x), -5*std(d), 5*std(d), 1_000)
mc_integral_x_1k = monte_carlo_integral(x -> x * pdf(d, x), -5*std(d), 5*std(d), 1_000)
mc_integral_1_1k = monte_carlo_integral(x -> pdf(d, x), -5*std(d), 5*std(d), 1_000)

println("\nResults with 1,000 draws:")
println("Integral of x^2 * f(x): ", mc_integral_x2_1k)
println("Integral of x * f(x): ", mc_integral_x_1k)
println("Integral of f(x): ", mc_integral_1_1k)

println("\nApproximation errors with 1,000 draws:")
println("Variance error: ", abs(mc_integral_x2_1k - true_variance))
println("Mean error: ", abs(mc_integral_x_1k - true_mean))
println("PDF integral error: ", abs(mc_integral_1_1k - true_integral))

println("\nComment on Monte Carlo integration:")
println("Monte Carlo integration provides good approximations, especially with a large number of draws.")
println("With 1,000,000 draws, the approximations are very close to the true values.")
println("Using only 1,000 draws still gives reasonable approximations, but with noticeably larger errors.")
println("This demonstrates the trade-off between computational cost and accuracy in Monte Carlo methods.")
println("Increasing the number of draws improves accuracy but requires more computation time.")
######For question4 we need to use Gauss-Legendre quadrature to approximate the integral
include("lgwt.jl")
function mixed_logit_with_Z(theta, X, Z, y, nodes, weights)
    alpha = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    loglike = 0.0
    
    for i in 1:N
        prob_i = 0.0
        for (node, weight) in zip(nodes, weights)
            gamma = mu_gamma + sigma_gamma * node
            num = zeros(T,J)
            dem = 0.0
            for j=1:J
                num[j] = exp(dot(X[i,:], bigAlpha[:,j]) + gamma*(Z[i,j] - Z[i,J]))
                dem += num[j]
            end
            P = num ./ dem
            prob_i += weight * prod(P .^ bigY[i,:])
        end
        loglike -= log(prob_i)
    end
    
    return loglike
end

function optimize_mixed_logit(X, Z, y)
    nodes, weights = lgwt(7, -4, 4)
    startvals = [2*rand(7*size(X,2)).-1; .1; .1]
    td = TwiceDifferentiable(theta -> mixed_logit_with_Z(theta, X, Z, y, nodes, weights), startvals; autodiff = :forward)
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    return theta_hat_mle_ad, theta_hat_mle_ad_se
end
#####For the Monte Carlo integration which is an alternative to quadrature(Question5)
function mixed_logit_MC(theta, X, Z, y, D)
    alpha = theta[1:end-2]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = size(X,1)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    loglike = 0.0
    
    for i in 1:N
        prob_i = 0.0
        for _ in 1:D
            gamma = mu_gamma + sigma_gamma * randn()
            num = zeros(T,J)
            dem = 0.0
            for j=1:J
                num[j] = exp(dot(X[i,:], bigAlpha[:,j]) + gamma*(Z[i,j] - Z[i,J]))
                dem += num[j]
            end
            P = num ./ dem
            prob_i += prod(P .^ bigY[i,:])
        end
        loglike -= log(prob_i / D)
    end
    
    return loglike
end

function optimize_mixed_logit_MC(X, Z, y, D)
    startvals = [2*rand(7*size(X,2)).-1; .1; .1]
    td = TwiceDifferentiable(theta -> mixed_logit_MC(theta, X, Z, y, D), startvals; autodiff = :forward)
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    return theta_hat_mle_ad, theta_hat_mle_ad_se
end
######Wrap All the functions
function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df, X, Z, y = load_data(url)
    
    println("Estimating multinomial logit...")
    mlogit_theta, mlogit_se = optimize_mlogit(X, Z, y)
    println("Multinomial logit estimates:")
    println([mlogit_theta mlogit_se])
    
    println("\nEstimating mixed logit with quadrature...")
    mixed_theta, mixed_se = optimize_mixed_logit(X, Z, y)
    println("Mixed logit estimates (quadrature):")
    println([mixed_theta mixed_se])
    
    println("\nEstimating mixed logit with Monte Carlo...")
    mixed_mc_theta, mixed_mc_se = optimize_mixed_logit_MC(X, Z, y, 1000)
    println("Mixed logit estimates (Monte Carlo):")
    println([mixed_mc_theta mixed_mc_se])
end
###Test the functions
main()
using Test

@testset "Problem Set 4 Tests" begin
    # Test data loading
    @test begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
        df, X, Z, y = load_data(url)
        !isempty(df) && size(X, 2) == 3 && size(Z, 2) == 8 && length(y) == size(X, 1)
    end

    # Test mlogit_with_Z function
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(22)
        result = mlogit_with_Z(theta_test, X_test, Z_test, y_test)
        typeof(result) <: Real && !isnan(result)
    end

    # Test mixed_logit_with_Z function
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(23)
        nodes, weights = lgwt(7, -4, 4)
        result = mixed_logit_with_Z(theta_test, X_test, Z_test, y_test, nodes, weights)
        typeof(result) <: Real && !isnan(result)
    end

    # Test mixed_logit_MC function
    @test begin
        X_test = rand(100, 3)
        Z_test = rand(100, 8)
        y_test = rand(1:8, 100)
        theta_test = rand(23)
        D = 100
        result = mixed_logit_MC(theta_test, X_test, Z_test, y_test, D)
        typeof(result) <: Real && !isnan(result)
    end
end