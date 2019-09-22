using SteadyState
using Test
using QuantumOptics, SparseArrays
using IterativeSolvers

@testset "SteadyState.jl" begin
    ωc = 1.2
    ωa = 0.9
    g = 1.0
    γ = 0.5
    κ = 1.1

    T = Float64[0.,1.]


    fockbasis = FockBasis(10)
    spinbasis = SpinBasis(1//2)
    basis = tensor(spinbasis, fockbasis)

    sx = sigmax(spinbasis)
    sy = sigmay(spinbasis)
    sz = sigmaz(spinbasis)
    sp = sigmap(spinbasis)
    sm = sigmam(spinbasis)

    Ha = embed(basis, 1, 0.5*ωa*sz)
    Hc = embed(basis, 2, ωc*number(fockbasis))
    Hint = sm ⊗ create(fockbasis) + sp ⊗ destroy(fockbasis)
    H = Ha + Hc + Hint

    Ja_unscaled = embed(basis, 1, sm)
    Jc_unscaled = embed(basis, 2, destroy(fockbasis))
    Junscaled = [Ja_unscaled, Jc_unscaled]

    Ja = embed(basis, 1, sqrt(γ)*sm)
    Jc = embed(basis, 2, sqrt(κ)*destroy(fockbasis))
    J = [Ja, Jc]
    Jlazy = [LazyTensor(basis, 1, sqrt(γ)*sm), Jc]

    Hnh = H - 0.5im*sum([dagger(J[i])*J[i] for i=1:length(J)])

    Hdense = dense(H)
    Hlazy = LazySum(Ha, Hc, Hint)
    Hnh_dense = dense(Hnh)
    Junscaled_dense = map(dense, Junscaled)
    Jdense = map(dense, J)

    Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 5)
    ρ₀ = dm(Ψ₀)

    tout, ρt = timeevolution.master([0,100], ρ₀, Hdense, Jdense; reltol=1e-7)

    # Test defaults
    ρ1 = SteadyState.iterative(ρ₀, Hdense, Jdense; tol=1e-7)
    @test tracedistance(ρ1, ρt[end]) < 1e-6
    ρ1_bis = SteadyState.iterative(Hdense, Jdense; tol=1e-7)
    @test tracedistance(ρ1, ρ1_bis) < 1e-6

    ρ2 = SteadyState.iterative(ρ₀, H, J; tol=1e-7)
    @test tracedistance(DenseOperator(ρ2), ρt[end]) < 1e-6
    ρ2_bis = SteadyState.iterative(H, J; tol=1e-7)
    @test tracedistance(DenseOperator(ρ2), DenseOperator(ρ2_bis)) < 1e-6

    ρ3 = SteadyState.iterative(Ψ₀, Hdense, Jdense; tol=1e-7)
    @test tracedistance(ρ3, ρt[end]) < 1e-6
    ρ3_bis = SteadyState.iterative(Hdense, Jdense; tol=1e-7)
    @test tracedistance(ρ3, ρ3_bis) < 1e-6

    ρ4 = SteadyState.iterative(Ψ₀, H, J; tol=1e-7)
    @test tracedistance(DenseOperator(ρ4), ρt[end]) < 1e-6
    ρ4_bis = SteadyState.iterative(H, J; tol=1e-7)
    @test tracedistance(DenseOperator(ρ4), DenseOperator(ρ4_bis)) < 1e-6

    # Test explicit call to iterative solvers
    ρ1, ch = SteadyState.iterative(ρ₀, Hdense, Jdense, bicgstabl!, 4; log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(ρ1, ρt[end]) < 1e-6
    ρ1_bis, ch = SteadyState.iterative(Hdense, Jdense, bicgstabl!, 4; log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(ρ1, ρ1_bis) < 1e-6

    ρ2, ch = SteadyState.iterative(ρ₀, H, J, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(DenseOperator(ρ2), ρt[end]) < 1e-6
    ρ2_bis, ch = SteadyState.iterative(H, J, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(DenseOperator(ρ2), DenseOperator(ρ2_bis)) < 1e-6

    ρ3, ch = SteadyState.iterative(Ψ₀, Hdense, Jdense, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(ρ3, ρt[end]) < 1e-6
    ρ3_bis, ch = SteadyState.iterative(Hdense, Jdense, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(ρ3, ρ3_bis) < 1e-6

    ρ4, ch = SteadyState.iterative(Ψ₀, H, J, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(DenseOperator(ρ4), ρt[end]) < 1e-6
    ρ4_bis, ch = SteadyState.iterative(H, J, idrs!; s=15, log=true, tol=1e-7)
    @test ch.isconverged
    @test tracedistance(DenseOperator(ρ4), DenseOperator(ρ4_bis)) < 1e-6
end
