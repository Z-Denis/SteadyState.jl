using SteadyState
using Test
using QuantumOptics, SparseArrays
using IterativeSolvers

@testset "SteadyState.jl" begin
    b = GenericBasis(15)
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

    r0 = SparseOperator(b)
    r0.data[1,1] = one(ComplexF64)

    # Dense
    r3 = steadystate_iterative!(deepcopy(DenseOperator(r0)), DenseOperator(H), DenseOperator.(J), bicgstabl!, 2; tol=tol)
    @test tr(r3) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r3.data .- r3.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r3.data .- r.data')) ≈ 0.0 atol=tol

    #Sparse
    #=
    r4 = steadystate_iterative!(deepcopy(r0), H, J, :bicgstabl!, 2; tol=tol)
    @test tr(r4) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r4.data .- r4.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r4.data .- r.data')) ≈ 0.0 atol=tol
    =#

    # Dense
    r5 = steadystate_iterative!(deepcopy(DenseOperator(r0)), DenseOperator(H), DenseOperator.(J), gmres!; tol=tol)
    @test tr(r5) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r5.data .- r5.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r5.data .- r.data')) ≈ 0.0 atol=tol

    # Dense
    r6 = steadystate_iterative!(deepcopy(DenseOperator(r0)), DenseOperator(H), DenseOperator.(J), idrs!; s=4, tol=tol)
    @test tr(r6) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r6.data .- r6.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r6.data .- r.data')) ≈ 0.0 atol=tol

    # Sparse
    r7 = steadystate_iterative!(deepcopy(r0), H, J, idrs!; s=4, tol=tol)
    @test tr(r7) ≈ one(ComplexF64) atol=tol
    @test sum(abs2.(r7.data .- r7.data')) ≈ 0.0 atol=tol
    @test sum(abs2.(r7.data .- r.data')) ≈ 0.0 atol=tol
end
