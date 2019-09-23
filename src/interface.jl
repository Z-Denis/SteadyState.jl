"""
    iterative!(rho0, H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `rho0`: Initial density matrix.
* `H`: Non-lazy operator or arbitrary matrix specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be non-lazy operators types or matrices.
* `method!::Function`: The iterative method to be used. Defaults to
`IterativeSolvers.bicgstabl!` or `IterativeSolvers.idrs!` depending on arguments' types.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.

See also: [`iterative`](@ref)
"""
function iterative!(rho0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...;
                    rates::DecayRates=nothing, Jdagger::TJd=nothing, kwargs...) where {B<:Basis,TJd<:Union{Vector,Nothing}}
    _check_jump_ops(B,rates,J,Jdagger)

    sol = if Jdagger===nothing
        iterative!(rho0.data,H.data,map(x->x.data,J),method!,args...;rates=rates,Jdagger=nothing,kwargs...)
    else
        iterative!(rho0.data,H.data,map(x->x.data,J),method!,args...;rates=rates,Jdagger=map(x->x.data,Jdagger),kwargs...)
    end
    if typeof(sol) <: Tuple
        rho = deepcopy(H)
        rho.data .= sol[1]
        return rho, sol[2]
    else
        rho = deepcopy(H)
        rho.data .= sol
        return rho
    end
end

"""
    iterative(H, J, [method!], args...; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via an iterative method provided as argument.

# Arguments
* `H`: Non-lazy operator or arbitrary matrix specifying the Hamiltonian.
* `J`: Vector containing all jump operators which can be non-lazy operators types or matrices.
* `method!::Function`: The iterative method to be used. Defaults to
`IterativeSolvers.bicgstabl!` or `IterativeSolvers.idrs!` depending on arguments' types.
* `args...`: Further arguments are passed on to the iterative solver.
* `kwargs...`: Further keyword arguments are passed on to the iterative solver.

See also: [`iterative`](@ref)
"""
function iterative(H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...;
                   rates::DecayRates=nothing, Jdagger::TJd=nothing, kwargs...) where {B<:Basis,TJd<:Union{Vector,Nothing}}
    _check_jump_ops(B,rates,J,Jdagger)

    sol = if Jdagger===nothing
        iterative(H.data,map(x->x.data,J),method!,args...;rates=rates,Jdagger=nothing,kwargs...)
    else
        iterative(H.data,map(x->x.data,J),method!,args...;rates=rates,Jdagger=map(x->x.data,Jdagger),kwargs...)
    end
    if typeof(sol) <: Tuple
        rho = deepcopy(H)
        rho.data .= sol[1]
        return rho, sol[2]
    else
        rho = deepcopy(H)
        rho.data .= sol
        return rho
    end
end

iterative(rho0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...; rates::DecayRates=nothing, Jdagger::TJd=nothing, kwargs...) where {B<:Basis,TJd<:Union{Vector,Nothing}} = iterative!(copy(rho0),H,J,method!,args...;kwargs...)
iterative(psi0::Ket{B}, H::AbstractOperator{B,B}, J::Vector, method!::Union{Function,Missing}=missing, args...; rates::DecayRates=nothing, Jdagger::TJd=nothing, kwargs...) where {B<:Basis,TJd<:Union{Vector,Nothing}} = iterative!(dm(psi0),H,J,method!,args...;kwargs...)


function iterative!(rho0::AbstractMatrix{T1}, H::AbstractMatrix{T2}, J::Vector{TJ}, method!::Union{Function,Missing}=missing, args...;
                   rates::DecayRates=nothing, Jdagger::TJd=nothing, kwargs...) where {T1<:Number,T2<:Number,TJ<:AbstractMatrix,TJd<:Union{Vector{TJ},Nothing}}
    _check_jump_ops(rates,J,Jdagger)
    if ismissing(method!)
        if isblascompatible(rho0) && isblascompatible(H) && all(isblascompatible.(J))
            return steadystate_iterative!(rho0,H,rates,J,Jdagger,bicgstabl!,args...;kwargs...)
        else
            return steadystate_iterative!(rho0,H,rates,J,Jdagger,idrs!,args...;kwargs...)
        end
    else
        @assert eltype(J) <: AbstractMatrix{T} where {T<:Number} "The jump operators must be matrices of numbers."
        return steadystate_iterative!(rho0,H,rates,J,Jdagger,method!,args...;kwargs...)
    end
end

function iterative(H::AbstractMatrix{T}, J::Vector, method!::Union{Function,Missing}=missing, args...; kwargs...) where {T<:Number}
    rho0 = similar(H)
    rho0 .= zero(T)
    rho0[1,1] = one(T)
    return iterative!(rho0,H,J,method!,args...;kwargs...)
end

function _check_jump_ops(B::DataType, rates::DecayRates, J::Vector, Jdagger::Union{Vector,Nothing})
    @assert all(typeof(j) <: AbstractOperator{B,B} for j in J) "Jump operators have incompatible bases."
    if Jdagger!==nothing
        @assert all(typeof(j) <: AbstractOperator{B,B} for j in Jdagger) "Adjoint jump operators have incompatible bases."
    end
end

function _check_jump_ops(rates::DecayRates, J::Vector, Jdagger::Union{Vector,Nothing})
    @assert all(typeof(j) <: AbstractMatrix for j in J) "Jump operators must all be of the same type."
    if Jdagger!==nothing
        @assert all(typeof(j)==typeof(first(J)) for j in Jdagger) "Adjoint jump operators must all be of the same type."
    end
    if rates!==nothing
        if typeof(rates) <: Vector
            @assert length(rates) == length(J) "The number of rates must match that of jump operators."
            if Jdagger!==nothing
                @assert length(J) == length(Jdagger) "The number of jump operators must match that of adjoint jump operators."
            end
        else # rates <: Matrix
            @assert size(rates,1) == length(J) "The number of jump operators must be compatible with the rates."
            if Jdagger!==nothing
                @assert size(rates,2) == length(Jdagger) "The number of adjoint jump operators must be compatible with the rates."
            end
        end
    else # rates = nothing
        if Jdagger!==nothing
            @assert length(J) == length(Jdagger) "The number of jump operators must match that of adjoint jump operators."
        end
    end
end
