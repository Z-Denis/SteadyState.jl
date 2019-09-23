module SteadyState

using QuantumOptics
using IterativeSolvers, LinearMaps, LinearAlgebra, SparseArrays

const T_blas = Union{Float64,Float32,ComplexF64,ComplexF32}
const DecayRates = Union{Vector{T} where T<:Number, Matrix{T} where T<:Number, Nothing}

isblascompatible(M::Array{T,2}) where {T<:T_blas} = true
isblascompatible(M::AbstractArray{T,2}) where T = false

include("interface.jl")
include("generic_method.jl")

end # module
