# Install necessary packages
import Pkg
Pkg.add("CSV")
Pkg.add("JLD2")
Pkg.add("HDF5")
Pkg.add("DataFrames")
Pkg.add("Random")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("Distributions")

# Load the necessary packages
using CSV
using JLD2  # Ensure JLD2 is loaded
using HDF5  # Load HDF5 as an alternative
using DataFrames
using Random
using LinearAlgebra
using Statistics
using Distributions  # Load the Distributions package for Uniform and Normal distributions

# Function for Question 1
function q1()
    # Set the seed
    Random.seed!(1234)
    
    # (a) Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    
    # Matrix C: first 5 rows and first 5 columns of A, last two columns and first 5 rows of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    
    # Matrix D: A if A ≤ 0, otherwise 0
    D = [A[i, j] <= 0 ? A[i, j] : 0 for i in 1:size(A, 1), j in 1:size(A, 2)]
    
    # (b) Number of elements in A
    println("Number of elements in A: ", length(A))
    
    # (c) Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))
    
    # (d) Reshape B into a vector (vec operator)
    E = reshape(B, :)
    
    # (e) Create a 3D array F containing A and B in the third dimension
    F = cat(A, B, dims=3)
    
    # (f) Permute F from 10x7x2 to 2x10x7
    F_new = permutedims(F, (3, 1, 2))
    
    # (g) Kronecker product of B and C to create G
    G = kron(B, C)

    # Delete existing HDF5 file to avoid "already exists" error
    try
        rm("matrixpractice.h5")
    catch e
        println("No existing HDF5 file to delete, continuing...")
    end
    
    # Part (i) - Save matrices A, B, C, D to a single .jld2 file
    try
        JLD2.save("firstmatrix.jld2", "A" => A, "B" => B, "C" => C, "D" => D)
        println("Saved matrices A, B, C, D to firstmatrix.jld2 successfully")
    catch e
        println("Error saving matrices to firstmatrix.jld2: ", e)
    end

    # Part (j) - Export matrix C as a .csv file
    try
        CSV.write("Cmatrix.csv", DataFrame(C, :auto))
        println("Exported matrix C to Cmatrix.csv successfully")
    catch e
        println("Error exporting matrix C to Cmatrix.csv: ", e)
    end

    # Part (k) - Export matrix D as a tab-delimited .dat file
    try
        CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')
        println("Exported matrix D to Dmatrix.dat successfully")
    catch e
        println("Error exporting matrix D to Dmatrix.dat: ", e)
    end
    
    # Alternative: Save matrices using HDF5 if JLD2 fails
    println("Attempting to save using HDF5 as a fallback...")

    # Convert D to Float64 to avoid method error with Real type
    D_float = Float64.(D)
    
    try
        h5write("matrixpractice.h5", "A", A)
        println("Saved A using HDF5 successfully")
    catch e
        println("Error saving A using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "B", B)
        println("Saved B using HDF5 successfully")
    catch e
        println("Error saving B using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "C", C)
        println("Saved C using HDF5 successfully")
    catch e
        println("Error saving C using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "D", D_float)
        println("Saved D using HDF5 successfully")
    catch e
        println("Error saving D using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "E", E)
        println("Saved E using HDF5 successfully")
    catch e
        println("Error saving E using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "F", F)
        println("Saved F using HDF5 successfully")
    catch e
        println("Error saving F using HDF5: ", e)
    end

    try
        h5write("matrixpractice.h5", "G", G)
        println("Saved G using HDF5 successfully")
    catch e
        println("Error saving G using HDF5: ", e)
    end
    
    # Return matrices
    return A, B, C, D
end

# Call the function q1()
A, B, C, D = q1()


# Install necessary packages
# Install necessary packages
import Pkg
Pkg.add("Distributions")

# Load the necessary packages
using Random
using Statistics
using Distributions
using LinearAlgebra

function q2(A, B, C)
    # (a) Element-wise product using a comprehension
    AB = [A[i, j] * B[i, j] for i in 1:size(A, 1), j in 1:size(A, 2)]
    
    # (a) Without a loop (element-wise multiplication using `.*`)
    AB2 = A .* B

    # (b) Using a loop to create Cprime with elements between -5 and 5
    Cprime = []
    for i in 1:size(C, 1)
        for j in 1:size(C, 2)
            if -5 <= C[i, j] <= 5
                push!(Cprime, C[i, j])
            end
        end
    end
    
    # (b) Without a loop (using comprehensions)
    Cprime2 = [C[i, j] for i in 1:size(C, 1), j in 1:size(C, 2) if -5 <= C[i, j] <= 5]

    # (c) Create a 3D array X with dimensions N×K×T
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    
    for t in 1:T
        X[:, 1, t] = ones(N)  # Intercept
        X[:, 2, t] = rand(Binomial(1, 0.75 * (6 - t) / 5), N)  # Dummy variable
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)  # Continuous variable
        X[:, 4, t] = rand(Normal(pi * (6 - t) / 3, 1 / exp(1)), N)  # Continuous variable
        X[:, 5, t] = rand(Binomial(20, 0.6), N)  # Discrete normal variable
        X[:, 6, t] = rand(Binomial(20, 0.5), N)  # Binomial variable
    end

    # (d) Create a matrix β with dimensions K×T
    β = vcat(
        [1 + 0.25 * (t - 1) for t in 1:T],  # Row 1
        [log(t) for t in 1:T],  # Row 2
        [-sqrt(t) for t in 1:T],  # Row 3
        [exp(t) - exp(t + 1) for t in 1:T],  # Row 4
        [t for t in 1:T],  # Row 5
        [t / 3 for t in 1:T]  # Row 6
    )
    
    # Convert the list of vectors into a matrix of size K×T
    β = reshape(β, K, T)

    # (e) Create a matrix Y with dimensions N×T
    Y = Array{Float64}(undef, N, T)
    σ = 0.36  # Standard deviation for εt

    for t in 1:T
        εt = rand(Normal(0, σ), N)
        # Ensure dimensions are consistent for matrix multiplication
        Y[:, t] = X[:, :, t] * β[:, t] + εt
    end
    
    println("Completed question 2")
    
    # Return the matrices and vectors created
    return AB, AB2, Cprime, Cprime2, X, β, Y
end

# Example call to the function:
# A, B, C = some_matrices_defined_earlier()
# q2(A, B, C)
q2(A, B, C)
# Install necessary packages
import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("FreqTables")
using CSV
using DataFrames
using FreqTables

# Main function for Question 3
function q3()
    file_path = "c:/Users/ASUSCenter/fall-2024/ProblemSets/PS1-julia-intro/nlsw88.csv"
    
    # Check if the file exists before attempting to read it
    if !isfile(file_path)
        error("File $file_path does not exist. Please check the file path.")
    end
    
    try
        # Attempt to read the CSV file using CSV.jl
        nlsw88 = CSV.read(file_path, DataFrame; delim=",", ignorerepeated=true, quotechar='"', missingstring="NA", normalizenames=true)
        println("Successfully imported and processed `nlsw88.csv`.")
    catch e
        println("Error reading the CSV file: ", e)
        return
    end
    
    # Print column names to inspect them
    println("Column names in the DataFrame: ", names(nlsw88))
    
    # (b) Calculate percentages for never married and college graduates
    never_married_percentage = sum(nlsw88[!, :married] .== "0") / nrow(nlsw88) * 100
    college_graduate_percentage = sum(nlsw88[!, :collgrad] .== "1") / nrow(nlsw88) * 100
    println("Percentage never married: ", never_married_percentage, "%")
    println("Percentage college graduates: ", college_graduate_percentage, "%")
    
    # (c) Race category percentage
    race_freq_table = freqtable(nlsw88[!, :race])
    race_percentage_table = 100 * race_freq_table / nrow(nlsw88)
    println("Percentage by race:")
    println(race_percentage_table)
    
    # (d) Summary statistics
    summarystats = describe(nlsw88)
    missing_grade_count = sum(ismissing.(nlsw88[!, :grade]))
    
    # Print summary statistics and the number of missing grade observations
    println("Summary statistics:")
    println(summarystats)
    println("Number of missing grade observations: ", missing_grade_count)
    
    # (e) Joint distribution of industry and occupation
    joint_dist = combine(groupby(nlsw88, [:industry, :occupation]), nrow => :count)
    println("Joint distribution of industry and occupation:")
    println(joint_dist)
    
    # (f) Mean wage by industry and occupation
    subset_df = select(nlsw88, :industry, :occupation, :wage)
    mean_wage = combine(groupby(subset_df, [:industry, :occupation]), :wage => mean => :mean_wage)
    println("Mean wage by industry and occupation:")
    println(mean_wage)
    
    println("Completed question 3.")
    
    return nlsw88, never_married_percentage, college_graduate_percentage, race_percentage_table, summarystats, missing_grade_count, joint_dist, mean_wage
end

# Run the main function for question 3
q3()
using JLD2, CSV, DataFrames, LinearAlgebra
using JLD2, Random, Distributions

println("Current working directory: ", pwd())
using JLD2, Random, Distributions

# Set the full path for saving the file
file_path = "c:/Users/ASUSCenter/fall-2024/ProblemSets/PS1-julia-intro/firstmatrix.jld2"

# Function to create and save matrices (from Question 1)
function save_matrices(file_path)
    # Set the seed
    Random.seed!(1234)
    
    # Create matrices A and B
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    
    # Matrix C: first 5 rows and first 5 columns of A, last two columns and first 5 rows of B
    C = hcat(A[1:5, 1:5], B[1:5, 6:7])
    
    # Matrix D: A if A ≤ 0, otherwise 0
    D = [A[i, j] <= 0 ? A[i, j] : 0 for i in 1:size(A, 1), j in 1:size(A, 2)]
    
    # Save matrices A, B, C, D to a JLD2 file with the full path
    JLD2.@save file_path A B C D
    println("Matrices A, B, C, and D have been saved to '$file_path'.")
end

# Run the save_matrices function
save_matrices(file_path)



# Function for matrix operations
using JLD2, CSV, DataFrames, LinearAlgebra

# Function for matrix operations with debug information
# (i) Element-wise product
    elementwise_product = A .* B

    # (ii) Matrix product A'B (A transposed multiplied by B)
    matrix_product = A' * B

    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)

    return elementwise_product, matrix_product, sum_elements


# Main function for Question 4
function q4()
    # Specify the full path to the JLD2 file
    file_path = "c:/Users/ASUSCenter/fall-2024/ProblemSets/PS1-julia-intro/firstmatrix.jld2"
    
    try
        # Load matrices from the specified JLD2 file
        firstmatrix_data = JLD2.load(file_path)
        A = firstmatrix_data["A"]
        B = firstmatrix_data["B"]
        C = firstmatrix_data["C"]
        D = firstmatrix_data["D"]
        println("Matrices A, B, C, and D loaded from '$file_path'")
    catch e
        println("Error loading matrices from '$file_path': ", e)
        return
    end
    
    # (d) Evaluate matrixops() using A and B
    elementwise_product_AB, matrix_product_AB, sum_elements_AB = matrixops(A, B)
    println("Matrix operations on A and B:")
    println("Element-wise product of A and B:\n", elementwise_product_AB)
    println("Matrix product of A'B:\n", matrix_product_AB)
    println("Sum of all elements in A + B: ", sum_elements_AB)
    
    # (f) Evaluate matrixops() using C and D
    elementwise_product_CD, matrix_product_CD, sum_elements_CD = matrixops(C, D)
    println("Matrix operations on C and D:")
    println("Element-wise product of C and D:\n", elementwise_product_CD)
    println("Matrix product of C'D:\n", matrix_product_CD)
    println("Sum of all elements in C + D: ", sum_elements_CD)
    
    # (g) Evaluate matrixops() using ttl_exp and wage from nlsw88_processed.csv
    try
        nlsw88_processed = CSV.read("nlsw88_processed.csv", DataFrame)
        ttl_exp = convert(Array, nlsw88_processed.ttl_exp)
        wage = convert(Array, nlsw88_processed.wage)
        
        elementwise_product_exp_wage, matrix_product_exp_wage, sum_elements_exp_wage = matrixops(ttl_exp, wage)
        println("Matrix operations on ttl_exp and wage:")
        println("Element-wise product of ttl_exp and wage:\n", elementwise_product_exp_wage)
        println("Matrix product of ttl_exp'wage:\n", matrix_product_exp_wage)
        println("Sum of all elements in ttl_exp + wage: ", sum_elements_exp_wage)
    catch e
        println("Error processing ttl_exp and wage from nlsw88_processed.csv: ", e)
    end
    
    println("Completed question 4.")
end 
# Call the q4 function to run the operations
q4()
using Test, JLD2, Random, Distributions, LinearAlgebra, CSV, DataFrames
cd("c:/Users/ASUSCenter/fall-2024/ProblemSets/PS1-julia-intro/")


# Include the functions to be tested (q1, q2, q3, q4, save_matrices, matrixops, etc.)
include("ps1.jl")  # Assuming your functions are in a file called ps1.jl

# Helper function to check if the CSV file exists
function check_csv_file_exists(file_path)
    if !isfile(file_path)
        error("File $file_path does not exist. Please check the file path.")
    end
end

# Unit tests
@testset "Problem Set 1 Tests" begin

    # Test for Question 1
    @testset "q1 function" begin
        A, B, C, D = q1()
        
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
        
        # Verify the content of the matrices (basic checks)
        @test all(D .<= 0)  # D should have only non-positive elements
    end

    # Test for Question 2
    @testset "q2 function" begin
        A, B, C = rand(Uniform(-5, 10), 10, 7), rand(Normal(-2, 15), 10, 7), rand(Uniform(-5, 5), 10, 7)
        AB, AB2, Cprime, Cprime2, X, β, Y = q2(A, B, C)
        
        # Verify dimensions
        @test size(AB) == size(A)
        @test size(AB2) == size(A)
        @test size(Cprime2) == length(Cprime)
        @test size(X) == (15169, 6, 5)
        @test size(β) == (6, 5)
        @test size(Y) == (15169, 5)
    end

    # Test for Question 3
    @testset "q3 function" begin
        # Specify the correct file path for nlsw88.csv
        file_path = "c:/Users/ASUSCenter/fall-2024/ProblemSets/PS1-julia-intro/nlsw88.csv"
        
        # Check if the CSV file exists before running the q3 function
        check_csv_file_exists(file_path)
        
        # Run the q3 function
        nlsw88, never_married_percentage, college_graduate_percentage, race_percentage_table, summarystats, missing_grade_count, joint_dist, mean_wage = q3()
        
        # Basic checks on the data
        @test !isempty(nlsw88)
        @test isnumeric(never_married_percentage)
        @test isnumeric(college_graduate_percentage)
        @test missing_grade_count >= 0
    end

    # Test for Question 4
    @testset "q4 function" begin
        @test q4() === nothing  # q4 doesn't return anything, so we check that it runs without errors
    end

    # Test save_matrices function from Question 1
    @testset "save_matrices function" begin
        # Run the function to save matrices
        save_matrices()
        
        # Check if the file has been created
        @test isfile("firstmatrix.jld2")
        
        # Load the matrices to check their dimensions
        firstmatrix_data = JLD2.load("firstmatrix.jld2")
        A = firstmatrix_data["A"]
        B = firstmatrix_data["B"]
        C = firstmatrix_data["C"]
        D = firstmatrix_data["D"]
        
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)
    end

    # Test matrixops function from Question 4
    @testset "matrixops function" begin
        A = rand(Uniform(-5, 10), 10, 7)
        B = rand(Normal(-2, 15), 10, 7)
        
        # Test for matrices with the same size
        elementwise_product, matrix_product, sum_elements = matrixops(A, B)
        @test size(elementwise_product) == size(A)
        @test size(matrix_product) == (7, 7)  # A' * B results in a 7x7 matrix
        @test sum_elements == sum(A + B)
        
        # Test for matrices with different sizes (should throw an error)
        C = rand(Uniform(-5, 10), 5, 7)
        @test_throws ErrorException matrixops(C, B)
    end

end

# There is a simple problem with my cvs file that I couldn't fix. If there is any error in tests, it relates to that problem.

