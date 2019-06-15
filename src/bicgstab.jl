using IterativeSolvers, LinearMaps, LinearAlgebra

"""
    steadystate_bicg(H, J, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via the stabilized biconjugate gradient method.
The first line of the Liouvillian is overwritten to enforce a non-trivial trace
one solution, this approximation yields an error of the order of the inverse of
the square of the size of the Hilbert space.

# Arguments
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: Further arguments are passed on to the iterative solver.

See also: [`CornerSpaceRenorm.steadystate_bicg_LtL`](@ref)
"""
function steadystate_bicg(H::DenseOperator{B,B}, J::Vector{O}, l::Int=2; log::Bool=false, kwargs...) where {B<:Basis,O<:DenseOperator{B,B}}
    ρ0 = DenseOperator(H.basis_l)
    ρ0.data[1,1] = ComplexF64(1.0)
    return steadystate_bicg!(ρ0,H,J,l;log=log,kwargs...)
end

"""
    steadystate_bicg(s, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of an `AbstractSystem` by solving
`L rho = 0` via the stabilized biconjugate gradient method.
The first line of the Liouvillian is overwritten to enforce a non-trivial trace
one solution, this approximation yields an error of the order of the inverse of
the square of the size of the Hilbert space.

# Arguments
* `s`: Instance of any subtype of `AbstractSystem`.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: Further arguments are passed on to the iterative solver.

See also: [`CornerSpaceRenorm.steadystate_bicg_LtL`](@ref)
"""
steadystate_bicg(s::AbstractSystem, l::Int=2; log::Bool=false, kwargs...) = steadystate_bicg(DenseOperator(s.H), DenseOperator.(s.J), l; log=log, kwargs...)

"""
    steadystate_bicg!(rho0, H, J, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via the stabilized biconjugate gradient method.
The first line of the Liouvillian is overwritten to enforce a non-trivial trace
one solution, this approximation yields an error of the order of the inverse of
the square of the size of the Hilbert space.

# Arguments
* `rho0`: Initial guess.
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: Further arguments are passed on to the iterative solver.

See also: [`steadystate_bicg`](@ref)
"""
function steadystate_bicg!(ρ0::DenseOperator{B,B}, H::DenseOperator{B,B}, J::Vector{O}, l::Int=2; log::Bool=false, tol::Float64 = sqrt(eps(real(ComplexF64))), kwargs...) where {B<:Basis,O<:DenseOperator{B,B}}
    # Size of the Hilbert space
    M::Int = size(H.data,1)
    # Non-Hermitian Hamiltonian
    iHnh::Matrix{ComplexF64} = -im*H.data
    for i in 1:length(J)
        iHnh .+= -0.5adjoint(J[i].data)*J[i].data
    end

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    function mvecmul!(y::AbstractVector, x::AbstractVector)
        y .= zero(eltype(x));
        ym::Matrix{ComplexF64} = reshape(y,M,M)
        ρ::Matrix{ComplexF64} = reshape(x,M,M)
        Jρ_cache::Matrix{ComplexF64} = similar(ρ)

        BLAS.gemm!('N', 'N', one(ComplexF64), iHnh, ρ, one(ComplexF64), ym)
        BLAS.gemm!('N', 'C', one(ComplexF64), ρ, iHnh, one(ComplexF64), ym)
        @inbounds @views for i in 1:length(J)
            BLAS.gemm!('N','N', one(ComplexF64), J[i].data, ρ, zero(ComplexF64), Jρ_cache)
            BLAS.gemm!('N','C', one(ComplexF64), Jρ_cache, J[i].data, one(ComplexF64), ym)
        end
        y .= reshape(ym,size(y))
        y[1] = tr(ρ)

        return y
    end
    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0::Vector{ComplexF64} = reshape(ρ0.data,M^2)
    y::Vector{ComplexF64}  = zeros(ComplexF64,M^2)
    y[1] = one(ComplexF64)

    # Define the linear map lm: ρ ↦ L(ρ)
    lm = LinearMap{eltype(H.data)}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    # Perform the stabilized biconjugate gradient procedure and devectorize ρ
    res0_norm::Float64 = norm(mvecmul!(similar(y),x0) .- y)
    tol /= res0_norm + eps(Float64)
    if !log
        ρ0.data .= reshape(bicgstabl!(x0,lm,y,l;tol=tol,kwargs...),(M,M))
        return ρ0
    else
        R::Vector{ComplexF64}, history = bicgstabl!(x0,lm,y,l;log=log,tol=tol,kwargs...)
        ρ0.data .= reshape(R,(M,M))
        return ρ0, history
    end
end

"""
    steadystate_bicg!(rho0, s, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of an `AbstractSystem` by solving
`L rho = 0` via the stabilized biconjugate gradient method.
The first line of the Liouvillian is overwritten to enforce a non-trivial trace
one solution, this approximation yields an error of the order of the inverse of
the square of the size of the Hilbert space.

# Arguments
* `rho0`: Initial guess.
* `s`: Instance of any subtype of `AbstractSystem`.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: Further arguments are passed on to the iterative solver.

See also: [`steadystate_bicg`](@ref)
"""
steadystate_bicg!(ρ0::DenseOperator{B,B}, s::AbstractSystem, l::Int=2; log::Bool=false, tol::Float64 = sqrt(eps(real(ComplexF64))), kwargs...) where {B<:Basis} = steadystate_bicg!(ρ0, DenseOperator(s.H), DenseOperator.(s.J), l; log=log, tol=tol, kwargs...)
