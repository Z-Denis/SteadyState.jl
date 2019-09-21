const T_blas = Union{Float64,Float32,ComplexF64,ComplexF32}

isblascompatible(M::AbstractMatrix{T}) where {T<:Number} = typeof(M)<:DenseMatrix && eltype(M)<:T_blas

function iterative!(rho0::AbstractMatrix{T1}, H::AbstractMatrix{T2}, J::Vector, method!::Union{Function,Missing}, args...; kwargs...) where {T1<:Number,T2<:Number}
    if ismissing(method!)
        if isblascompatible(rho0) && isblascompatible(H) && isblascompatible(eltype(J))
            return steadystate_iterative!(rho0,H,J,bicgstabl!,args...;kwargs...)
        else
            return steadystate_iterative!(rho0,H,J,idrs!,args...;kwargs...)
        end
    else
        @assert eltype(J) <: AbstractMatrix{T} where {T<:Number} "The jump operators must be matrices of numbers."
        return steadystate_iterative!(rho0,H,J,method!,args...;kwargs...)
    end
end

function iterative(H::AbstractMatrix{T}, J::Vector, method!::Union{Function,Missing}, args...; kwargs...) where {T<:Number}
    rho0 = similar(H)
    rho0[1,1] = one(T)
    return iterative!(rho0,H,J,method!,args...;kwargs...)
end

function iterative!(rho0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}, args...; kwargs...) where {B<:Basis}
    @assert all(typeof(j) <: AbstractOperator{B,B} for j in J) "Jump operators have incompatible bases."
    # TODO: relax this condition
    @assert all(typeof(j)==first(J) for j in J) "Jump operators must all be of the same type."
    iterative!(rho0.data,H.data,map(x->x.data,J),method!,args...;kwargs...)
end

function iterative(H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}, args...; kwargs...) where {B<:Basis}
    @assert all(typeof(j) <: AbstractOperator{B,B} for j in J) "Jump operators have incompatible bases."
    # TODO: relax this condition
    @assert all(typeof(j)==first(J) for j in J) "Jump operators must all be of the same type."
    iterative(H.data,map(x->x.data,J),method!,args...;kwargs...)
end
