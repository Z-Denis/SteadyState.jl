"""
    steadystate_iterative!(rho0, H, J, method!, args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `rho0`: Initial guess.
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `method!::Function`: some iterative method.
* `args...`: further arguments are passed on to the iterative solver.
* `kwargs...`: further keyword arguments are passed on to the iterative solver.

See also: [`steadystate_bicg`](@ref)
"""
function steadystate_iterative!(ρ0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector{O}, method!::Function, args...; log::Bool=false, tol::Float64 = sqrt(eps(Float64)), kwargs...) where {B<:Basis,O<:AbstractOperator{B,B}}
    if !log
        steadystate_iterative!(ρ0.data, H.data, map(x->x.data, J), method!, args...; log=log, tol=tol, kwargs...)
        return ρ0
    else
        _, history = steadystate_iterative!(ρ0.data, H.data, map(x->x.data, J), method!, args...; log=log, tol=tol, kwargs...)
        return ρ0, history
    end
end

function steadystate_iterative!(ρ0::Tρ, H::TH, J::Vector{TJ}, method!::Function, args...;
                                log::Bool=false, tol::Float64 = sqrt(eps(Float64)), kwargs...) where {Tρ<:AbstractMatrix,TH<:AbstractMatrix,TJ<:AbstractMatrix}
    # Size of the Hilbert space
    M = size(H,1)
    # Non-Hermitian Hamiltonian
    iHnh = -im*H
    for Ji=J
        iHnh .+= -0.5Ji'Ji
    end
    Jρ_cache = similar(ρ0)

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    function mvecmul!(y::AbstractVector, x::AbstractVector)
        y .= zero(eltype(y));
        ym = Tρ(@views reshape(y[2:end], M, M))
        ρ  = Tρ(@views reshape(x[2:end], M, M))

        #println("===========")
        #println(typeof(iHnh), "\n", typeof(ρ), "\n", typeof(ym), "\n", eltype(J), "\n", typeof(Jρ_cache))
        #println("-----------")
        #println(typeof(iHnh'), "\n", typeof(first(J)'))
        #println("ping")
        ym .= iHnh * ρ .+ ρ *iHnh'
        for Ji=J
            Jρ_cache = Ji * ρ
            ym .+= Jρ_cache * Ji'
        end
        @views y[2:end] .= reshape(ym, M^2)
        #println("pong")
        y[1] = tr(ρ)

        return y
    end
    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0 = similar(ρ0, M^2+1)
    x0[1] = zero(x0[1])
    x0[2:end] .= reshape(ρ0, M^2)

    y = similar(ρ0, M^2+1)
    y .= zero(y)
    y[1] = one(y[1])

    # Define the linear map lm: ρ ↦ L(ρ)
    lm = LinearMap{eltype(iHnh)}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    # Perform the stabilized biconjugate gradient procedure and devectorize ρ
    res0_norm = norm(mvecmul!(similar(y),x0) .- y)
    tol /= res0_norm + eps(real(eltype(H)))
    if !log
        ρ0 .= @views reshape(method!(x0,lm,y,args...;tol=tol,kwargs...)[2:end],(M,M))
        return ρ0
    else
        R, history = method!(x0,lm,y,args...;log=log,tol=tol,kwargs...)
        ρ0 .= @views reshape(R[2:end],(M,M))
        return ρ0, history
    end
end

function steadystate_iterative!(ρ0::Tm, H::Tm, J::Vector{Tm}, method!::Function, args...; log::Bool=false, tol::Float64 = sqrt(eps(Float64)), kwargs...) where {T<:Union{Float64,Float32,ComplexF64,ComplexF32},Tm<:DenseArray{T,2}}
    # Size of the Hilbert space
    M = size(H,1)
    # Non-Hermitian Hamiltonian
    iHnh = -im*H
    for Ji=J
        iHnh .+= -0.5Ji'Ji
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
        for Ji=J
            BLAS.gemm!('N','N', one(eltype(y)), Ji, ρ, zero(eltype(y)), Jρ_cache)
            BLAS.gemm!('N','C', one(eltype(y)), Jρ_cache, Ji, one(eltype(y)), ym)
        end

        y[1] = tr(ρ)

        return y
    end
    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0 = similar(ρ0, M^2+1)
    x0[1] = zero(x0[1])
    x0[2:end] .= reshape(ρ0, M^2)

    y = similar(ρ0, M^2+1)
    y .= zero(y)
    y[1] = one(y[1])

    # Define the linear map lm: ρ ↦ L(ρ)
    lm = LinearMap{eltype(iHnh)}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    # Perform the stabilized biconjugate gradient procedure and devectorize ρ
    res0_norm = norm(mvecmul!(similar(y),x0) .- y)
    tol /= res0_norm + eps(real(eltype(H)))

    if !log
        ρ0 .= @views reshape(method!(x0,lm,y,args...;tol=tol,kwargs...)[2:end],(M,M))
        return ρ0
    else
        R, history = method!(x0,lm,y,args...;log=log,tol=tol,kwargs...)
        ρ0 .= @views reshape(R[2:end],(M,M))
        return ρ0, history
    end
end
