
function steadystate_iterative!(rho0::Trho, H::AbstractMatrix, J::Vector{TJ}, method!::Function, args...;kwargs...) where {Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    # Size of the Hilbert space
    M = size(H,1)
    # Non-Hermitian Hamiltonian
    iHnh = -im*H
    for Ji=J
        iHnh .+= -0.5Ji'Ji
    end
    Jrho_cache = similar(rho0)

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    mvecmul!(y::AbstractVector, x::AbstractVector) = residual!(y,x,iHnh,J,Jrho_cache)

    # Solution x must satisfy L.x = y with y[1] = tr(x) = 1 and y[j≠1] = 0.
    x0 = similar(rho0, M^2+1)
    x0[1] = zero(x0[1])
    x0[2:end] .= reshape(rho0, M^2)

    y = similar(rho0, M^2+1)
    y .= zero(y)
    y[1] = one(y[1])

    # Define the linear map lm: rho ↦ L(rho)
    lm = LinearMap{eltype(iHnh)}(mvecmul!, length(y)::Int, length(y)::Int; ismutating=true, issymmetric=false, ishermitian=false, isposdef=false)

    log = haskey(kwargs, :log) ? kwargs[:log] : false
    _kwargs = filter(arg -> arg[1]!=:log, kwargs)

    # Perform the stabilized biconjugate gradient procedure and devectorize rho
    if !log
        rho0 .= @views reshape(method!(x0,lm,y,args...;log=log,_kwargs...)[2:end],(M,M))
        return rho0
    else
        R, history = method!(x0,lm,y,args...;log=log,_kwargs...)
        rho0 .= @views reshape(R[2:end],(M,M))
        return rho0, history
    end
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, J::Vector{TJ}, Jrho_cache::Trho) where {Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    ym  = Trho(@views reshape(y[2:end], M, M))
    rho = Trho(@views reshape(x[2:end], M, M))

    ym .= iHnh * rho .+ rho *iHnh'
    for Ji=J
        Jrho_cache = Ji * rho
        ym .+= Jrho_cache * Ji'
    end
    @views y[2:end] .= reshape(ym, M^2)
    
    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, J::Vector{Tm}, Jrho_cache::Tm) where {T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    ym  = @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), ym)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), ym)
    for Ji=J
        BLAS.gemm!('N','N', one(eltype(y)), Ji, rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','C', one(eltype(y)), Jrho_cache, Ji, one(eltype(y)), ym)
    end

    y[1] = tr(rho)

    return y
end
