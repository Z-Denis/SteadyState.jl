using SteadyState
using Test
using QuantumOptics, SparseArrays

@testset "SteadyState.jl" begin
    b = GenericBasis(10)
    H = SparseOperator(b,sprand(ComplexF64,b.shape[],b.shape[],0.1))
    H = H + dagger(H)
    J = [SparseOperator(b,sprand(ComplexF64,b.shape[],b.shape[],0.1)) for i in 1:3]

    tol = 1e-9

    # Reference
    r = steadystate.eigenvector(H, J; tol=tol)

    # Dense
    r1 = steadystate_bicg(DenseOperator(H), DenseOperator.(J), 4; tol=tol)
    @test tr(r1) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r1.data .- r1.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r1.data .- r.data')) ≈ 0.0 atol=tol

    #Sparse
    #=
    r2 = steadystate_bicg(H, J, 4; tol=tol)
    @test tr(r2) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r2.data .- r2.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r2.data .- r.data')) ≈ 0.0 atol=tol
    =#
end
