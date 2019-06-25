
"""
    steadystate_bicg(H, J, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via the stabilized biconjugate gradient method.

# Arguments
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: further arguments are passed on to the iterative solver.

See also: [`steadystate_bicg!`](@ref)
"""
function steadystate_bicg(H::O, J::Vector{O}, l::Int=2; log::Bool=false, kwargs...) where {B<:Basis,O<:AbstractOperator{B,B}}
    ρ0 = O<:DenseOperator ? DenseOperator(H.basis_l) : SparseOperator(H.basis_l)
    ρ0.data[1,1] = ComplexF64(1.0)
    return steadystate_iterative!(ρ0,H,J,bicgstabl!,l;log=log,kwargs...)
end

"""
    steadystate_bicg!(rho0, H, J, l=2; [log=false], kwargs...) -> rho[, log]

Compute the steady state density matrix of a Hamiltonian and a set of jump
operators by solving `L rho = 0` via the stabilized biconjugate gradient method.

# Arguments
* `rho0`: Initial guess.
* `H`: dense Hamiltonian.
* `J`: array of dense jump operators.
* `l`: number of GMRES steps per iteration.
* `kwargs...`: further arguments are passed on to the iterative solver.

See also: [`steadystate_bicg`](@ref)
"""
steadystate_bicg!(ρ0::AbstractOperator{B,B}, H::AbstractOperator{B,B}, J::Vector{O}, l::Int=2; log::Bool=false, kwargs...) where {B<:Basis,O<:AbstractOperator{B,B}} = steadystate_iterative!(ρ0,H,J,bicgstabl!,l;log=log,kwargs...)
