"""
    steadystate_iterative!(rho0, H, J, solver, args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `rho0`: Initial guess.
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `solver::Symbol`: name of a desired iterative method from scope or `IterativeSolvers.jl`.
* `args...`: further arguments are passed on to the iterative solver.
* `kwargs...`: further keyword arguments are passed on to the iterative solver.

See also: [`steadystate_bicg`](@ref)
"""
function steadystate_iterative!(ρ0::DenseOperator{B,B}, H::DenseOperator{B,B}, J::Vector{O}, solver::Symbol, args...; log::Bool=false, tol::Float64 = sqrt(eps(real(ComplexF64))), kwargs...) where {B<:Basis,O<:DenseOperator{B,B}}
    method! = isdefined(Main,solver) ? getfield(Main,solver) : getfield(@__MODULE__,solver)
    jump_ops = map(x->x.data, J)
    ρ0_mat = ρ0.data

    # Size of the Hilbert space
    M = size(H.data,1)
    # Non-Hermitian Hamiltonian
    iHnh = -im*H.data
    for Jᵢ=jump_ops
        iHnh .+= -0.5adjoint(Jᵢ)*Jᵢ
    end
    Jρ_cache = similar(iHnh)

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    function mvecmul!(y::AbstractVector, x::AbstractVector)
        y .= zero(eltype(y));
        ym = @views reshape(y[2:end], M, M)
        ρ  = @views reshape(x[2:end], M, M)

        BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, ρ, one(eltype(y)), ym)
        BLAS.gemm!('N', 'C', one(eltype(y)), ρ, iHnh, one(eltype(y)), ym)
        for Jᵢ=jump_ops
            BLAS.gemm!('N','N', one(eltype(y)), Jᵢ, ρ, zero(eltype(y)), Jρ_cache)
            BLAS.gemm!('N','C', one(eltype(y)), Jρ_cache, Jᵢ, one(eltype(y)), ym)
        end
        y[1] = tr(ρ)

        return y
    end
    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0 = similar(ρ0_mat, M^2+1)
    x0[1] = zero(x0[1])
    x0[2:end] .= reshape(ρ0_mat, M^2)

    y = similar(ρ0_mat, M^2+1)
    y .= zero(y)
    y[1] = one(y[1])

    # Define the linear map lm: ρ ↦ L(ρ)
    lm = LinearMap{eltype(iHnh)}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    # Perform the stabilized biconjugate gradient procedure and devectorize ρ
    res0_norm = norm(mvecmul!(similar(y),x0) .- y)
    tol /= res0_norm + eps(Float64)
    if !log
        ρ0.data .= @views reshape(method!(x0,lm,y,args...;tol=tol,kwargs...)[2:end],(M,M))
        return ρ0
    else
        R, history = method!(x0,lm,y,args...;log=log,tol=tol,kwargs...)
        ρ0.data .= @views reshape(R[2:end],(M,M))
        return ρ0, history
    end
end

function steadystate_iterative!(ρ0::SparseOperator{B,B}, H::SparseOperator{B,B}, J::Vector{O}, solver::Symbol, args...; log::Bool=false, tol::Float64 = sqrt(eps(real(ComplexF64))), kwargs...) where {B<:Basis,O<:SparseOperator{B,B}}
    method! = isdefined(Main,solver) ? getfield(Main,solver) : getfield(@__MODULE__,solver)
    # Size of the Hilbert space
    M::Int = size(H.data,1)
    # Non-Hermitian Hamiltonian
    iHnh::SparseMatrixCSC{ComplexF64,Int} = -im*H.data
    for i in 1:length(J)
        iHnh .+= -0.5adjoint(J[i].data)*J[i].data
    end
    Jρ_cache::SparseMatrixCSC{ComplexF64,Int} = similar(iHnh)

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    function mvecmul!(y::AbstractVector, x::AbstractVector)
        y .= zero(eltype(y));
        ym::SparseMatrixCSC{eltype(y),Int} = @views reshape(y[2:end],M,M)
        ρ::SparseMatrixCSC{eltype(x),Int}  = @views reshape(x[2:end],M,M)

        ym .= iHnh * ρ .+ ρ * adjoint(iHnh)
        #ym .+= adjoint(ym)
        @inbounds @views for i in 1:length(J)
            Jρ_cache = J[i].data * ρ
            ym .+= Jρ_cache * adjoint(J[i].data)
        end
        @views y[2:end] .= reshape(ym,length(ym))
        y[1] = tr(ρ)

        return y
    end
    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0::SparseVector{ComplexF64,Int} = [zero(ComplexF64); reshape(ρ0.data,M^2)]
    y::SparseVector{ComplexF64,Int}  = [one(ComplexF64); zeros(ComplexF64,M^2)]

    # Define the linear map lm: ρ ↦ L(ρ)
    lm = LinearMap{ComplexF64}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    # Perform the stabilized biconjugate gradient procedure and devectorize ρ
    res0_norm::Float64 = norm(mvecmul!(similar(y),x0) .- y)
    tol /= res0_norm + eps(Float64)
    if !log
        ρ0.data .= reshape(method!(x0,lm,y,args...;tol=tol,kwargs...)[2:end],(M,M))
        return ρ0
    else
        R::Vector{ComplexF64}, history = method!(x0,lm,y,args...;log=log,tol=tol,kwargs...)
        ρ0.data .= reshape(R[2:end],(M,M))
        return ρ0, history
    end
end
