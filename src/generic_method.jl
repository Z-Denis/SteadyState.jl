
function steadystate_iterative!(rho0::Trho, H::AbstractMatrix, rates::DecayRates, J::Vector{TJ}, Jdagger::TJd, method!::Function, args...;kwargs...) where {Trho<:AbstractMatrix,TJ<:AbstractMatrix,TJd<:Union{Vector{TJ},Nothing}}
    # Size of the Hilbert space
    M = size(H,1)
    # Non-Hermitian Hamiltonian
    iHnh = nh_hamiltonian(H,rates,J,Jdagger)
    Jrho_cache = similar(rho0)

    # In-place update of y = Lx where L and x are respectively the vectorized
    # Liouvillian and the vectorized density matrix. y[1] is set to the trace
    # of the density matrix so as to enforce a trace one non-trivial solution.
    mvecmul!(y::AbstractVector, x::AbstractVector) = residual!(y,x,iHnh,nothing,J,nothing,Jrho_cache)

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

    # Perform the stabilized biconjugate gradient procedure and devectorize rho
    if !log
        rho0 .= @views reshape(method!(x0,lm,y,args...;kwargs...)[2:end],(M,M))
        return rho0
    else
        R, history = method!(x0,lm,y,args...;kwargs...)
        rho0 .= @views reshape(R[2:end],(M,M))
        return rho0, history
    end
end

function nh_hamiltonian(H::AbstractMatrix, rates::Nothing, J::Vector{T}, Jdagger::Nothing) where {T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J)
        iHnh .+= -0.5 .* J[i]'J[i]
    end
    return iHnh
end
function nh_hamiltonian(H::AbstractMatrix, rates::Vector{Q}, J::Vector{T}, Jdagger::Nothing) where {Q<:Number,T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J)
        iHnh .+= -0.5rates[i] .* J[i]'J[i]
    end
    return iHnh
end
function nh_hamiltonian(H::AbstractMatrix, rates::Matrix{Q}, J::Vector{T}, Jdagger::Nothing) where {Q<:Number,T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J), j=1:length(J)
        iHnh -= 0.5rates[i,j] .* J[i]'J[j]
    end
    return iHnh
end
function nh_hamiltonian(H::AbstractMatrix, rates::Nothing, J::Vector{T}, Jdagger::Vector{T}) where {T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J)
        iHnh .+= -0.5 .* Jdagger[i]*J[i]
    end
    return iHnh
end
function nh_hamiltonian(H::AbstractMatrix, rates::Vector{Q}, J::Vector{T}, Jdagger::Vector{T}) where {Q<:Number,T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J)
        iHnh .+= -0.5rates[i] .* Jdagger[i]*J[i]
    end
    return iHnh
end
function nh_hamiltonian(H::AbstractMatrix, rates::Matrix{Q}, J::Vector{T}, Jdagger::Vector{T}) where {Q<:Number,T<:AbstractMatrix}
    iHnh = -im*H
    for i=1:length(J), j=1:length(J)
        iHnh -= 0.5rates[i,j] .* Jdagger[i]*J[j]
    end
    return iHnh
end

# With Jdagger
function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Nothing, J::Vector{Tm}, Jdagger::Vector{Tm}, Jrho_cache::Tm) where {T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for i=1:length(J)
        BLAS.gemm!('N','N', one(eltype(y)), J[i], rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','N', one(eltype(y)), Jrho_cache, Jdagger[i], one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Vector{Q}, J::Vector{Tm}, Jdagger::Vector{Tm}, Jrho_cache::Tm) where {Q<:Number,T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for i=1:length(J)
        BLAS.gemm!('N','N', rates[i], J[i], rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','N', one(eltype(y)), Jrho_cache, Jdagger[i], one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Matrix{Q}, J::Vector{Tm}, Jdagger::Vector{Tm}, Jrho_cache::Tm) where {Q<:Number,T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for i=1:length(J), j=1:length(J)
        BLAS.gemm!('N','N', rates[i,j], J[i], rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','N', one(eltype(y)), Jrho_cache, Jdagger[j], one(eltype(y)), drho)

        BLAS.gemm!('N','N', -0.5one(eltype(y)), Jdagger[j], Jrho_cache, one(eltype(y)), drho)


        BLAS.gemm!('N','N', rates[i,j], rho, Jdagger[j], zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','N', -0.5one(eltype(y)), Jrho_cache, J[i], one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

# Without
function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Nothing, J::Vector{Tm}, Jdagger::Nothing, Jrho_cache::Tm) where {T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for Ji=J
        BLAS.gemm!('N','N', one(eltype(y)), Ji, rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','C', one(eltype(y)), Jrho_cache, Ji, one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Vector{Q}, J::Vector{Tm}, Jdagger::Nothing, Jrho_cache::Tm) where {Q<:Number,T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for i=1:length(J)
        BLAS.gemm!('N','N', rates[i], J[i], rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','C', one(eltype(y)), Jrho_cache, J[i], one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::Tm, rates::Matrix{Q}, J::Vector{Tm}, Jdagger::Nothing, Jrho_cache::Tm) where {Q<:Number,T<:T_blas,Tm<:DenseMatrix{T}}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    BLAS.gemm!('N', 'N', one(eltype(y)), iHnh, rho, one(eltype(y)), drho)
    BLAS.gemm!('N', 'C', one(eltype(y)), rho, iHnh, one(eltype(y)), drho)
    for i=1:length(J), j=1:length(J)
        BLAS.gemm!('N','N', rates[i,j], J[i], rho, zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','C', one(eltype(y)), Jrho_cache, J[j], one(eltype(y)), drho)

        BLAS.gemm!('C','N', -0.5one(eltype(y)), J[j], Jrho_cache, one(eltype(y)), drho)


        BLAS.gemm!('N','C', rates[i,j], rho, J[j], zero(eltype(y)), Jrho_cache)
        BLAS.gemm!('N','N', -0.5one(eltype(y)), Jrho_cache, J[i], one(eltype(y)), drho)
    end

    y[1] = tr(rho)

    return y
end

# No-BLAS fallback

# With Jdagger
function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Nothing, J::Vector{TJ}, Jdagger::Vector{TJ}, Jrho_cache::Trho) where {Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho  = Trho(@views reshape(y[2:end], M, M))
    rho = Trho(@views reshape(x[2:end], M, M))

    drho .= iHnh * rho .+ rho *iHnh'
    for i=1:length(J)
        Jrho_cache .= J[i] * rho
        drho .+= Jrho_cache * Jdagger[i]
    end
    @views y[2:end] .= reshape(drho, M^2)

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Vector{Q}, J::Vector{TJ}, Jdagger::Vector{TJ}, Jrho_cache::Trho) where {Q<:Number,Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    drho .= iHnh * rho .+ rho *iHnh'
    for i=1:length(J)
        Jrho_cache .= rates[i] .* J[i] * rho
        drho .+= Jrho_cache * Jdagger[i]
    end

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Matrix{Q}, J::Vector{TJ}, Jdagger::Vector{TJ}, Jrho_cache::Trho) where {Q<:Number,Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    drho .= iHnh * rho .+ rho *iHnh'
    for i=1:length(J), j=1:length(J)
        Jrho_cache .= rates[i,j] .* J[i] * rho
        drho .+= Jrho_cache * Jdagger[j]

        drho .+= -0.5 .* Jdagger[j] * Jrho_cache

        Jrho_cache .= rates[i,j] .* rho * Jdagger[j]
        drho .+= -0.5 .* Jrho_cache * J[i]
    end

    y[1] = tr(rho)

    return y
end

# Without
function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Nothing, J::Vector{TJ}, Jdagger::Nothing, Jrho_cache::Trho) where {Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho  = Trho(@views reshape(y[2:end], M, M))
    rho = Trho(@views reshape(x[2:end], M, M))

    drho .= iHnh * rho .+ rho *iHnh'
    for Ji=J
        Jrho_cache .= Ji * rho
        drho .+= Jrho_cache * Ji'
    end
    @views y[2:end] .= reshape(drho, M^2)

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Vector{Q}, J::Vector{TJ}, Jdagger::Nothing, Jrho_cache::Trho) where {Q<:Number,Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    drho .= iHnh * rho .+ rho *iHnh'
    for i=1:length(J)
        Jrho_cache .= rates[i] .* J[i] * rho
        drho .+= Jrho_cache * J[i]'
    end
    @views y[2:end] .= reshape(drho, M^2)

    y[1] = tr(rho)

    return y
end

function residual!(y::AbstractVector, x::AbstractVector, iHnh::AbstractMatrix, rates::Matrix{Q}, J::Vector{TJ}, Jdagger::Nothing, Jrho_cache::Trho) where {Q<:Number,Trho<:AbstractMatrix,TJ<:AbstractMatrix}
    M = size(iHnh,1)
    y  .= zero(eltype(y));
    drho= @views reshape(y[2:end], M, M)
    rho = @views reshape(x[2:end], M, M)

    drho .= iHnh * rho .+ rho *iHnh'
    for i=1:length(J), j=1:length(J)
        Jrho_cache .= rates[i,j] .* J[i] * rho
        drho .+= Jrho_cache * J[j]'

        drho .+= -0.5 .* J[j]' * Jrho_cache

        Jrho_cache .= rates[i,j] .* rho * J[j]'
        drho .+= -0.5 .* Jrho_cache * J[i]
    end

    y[1] = tr(rho)

    return y
end
