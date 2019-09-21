"""
    iterative!(rho0, H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `rho0`: Initial density operator.
* `H`: Arbitrary operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary operator type.
* `method!::Function`: The iterative method to be used. Defaults to
`IterativeSolvers.bicgstabl!` or `IterativeSolvers.idrs!` depending on arguments' types.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.

See also: [`iterative`](@ref)
"""
function iterative!(rho0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...; kwargs...) where {B<:Basis}
    @assert all(typeof(j) <: AbstractOperator{B,B} for j in J) "Jump operators have incompatible bases."
    # TODO: relax this condition
    @assert all(typeof(j)==typeof(first(J)) for j in J) "Jump operators must all be of the same type."
    iterative!(rho0.data,H.data,map(x->x.data,J),method!,args...;kwargs...)
end

"""
    iterative(H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `H`: Arbitrary operator specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be of any arbitrary operator type.
* `method!::Function`: The iterative method to be used. Defaults to
`IterativeSolvers.bicgstabl!` or `IterativeSolvers.idrs!` depending on arguments' types.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.

See also: [`iterative`](@ref)
"""
function iterative(H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...; kwargs...) where {B<:Basis}
    @assert all(typeof(j) <: AbstractOperator{B,B} for j in J) "Jump operators have incompatible bases."
    # TODO: relax this condition
    @assert all(typeof(j)==typeof(first(J)) for j in J) "Jump operators must all be of the same type."
    iterative(H.data,map(x->x.data,J),method!,args...;kwargs...)
end

function iterative!(rho0::AbstractMatrix{T1}, H::AbstractMatrix{T2}, J::Vector, method!::Union{Function,Missing}=missing, args...; kwargs...) where {T1<:Number,T2<:Number}
    if ismissing(method!)
        if isblascompatible(rho0) && isblascompatible(H) && all(isblascompatible.(J))
            return steadystate_iterative!(rho0,H,J,bicgstabl!,args...;kwargs...)
        else
            return steadystate_iterative!(rho0,H,J,idrs!,args...;kwargs...)
        end
    else
        @assert eltype(J) <: AbstractMatrix{T} where {T<:Number} "The jump operators must be matrices of numbers."
        return steadystate_iterative!(rho0,H,J,method!,args...;kwargs...)
    end
end

function iterative(H::AbstractMatrix{T}, J::Vector, method!::Union{Function,Missing}=missing, args...; kwargs...) where {T<:Number}
    rho0 = similar(H)
    rho0[1,1] = one(T)
    return iterative!(rho0,H,J,method!,args...;kwargs...)
end
