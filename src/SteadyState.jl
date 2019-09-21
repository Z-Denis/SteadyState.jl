module SteadyState

using QuantumOptics
using IterativeSolvers, LinearMaps, LinearAlgebra, SparseArrays

const T_blas = Union{Float64,Float32,ComplexF64,ComplexF32}
isblascompatible(M::AbstractMatrix{T}) where {T<:Number} = typeof(M)<:DenseMatrix && eltype(M)<:T_blas

include("interface.jl")
export iterative, iterative!

include("generic_method.jl")

end # module
