# SteadyState.jl

[![Build Status](https://travis-ci.com/Z-Denis/SteadyState.jl.svg?token=XuYcpCDomapYmd2vHj9y&branch=master)](https://travis-ci.com/Z-Denis/SteadyState.jl)
[![Codecov](https://codecov.io/gh/Z-Denis/SteadyState.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Z-Denis/SteadyState.jl)

Tiny package for determining iteratively the steady state of time-independent Liouville superoperators thanks to [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).

#### Important
The code was integrated into [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl/blob/master/src/steadystate_iterative.jl) and may be accessed directly via its [IPA](https://docs.qojulia.org/api/#QuantumOptics.steadystate.iterative).

#### Methods

```julia
rho = Steadystate.iterative([rho,] H, J[, method!], args...; kwargs...)
rho, log = Steadystate.iterative([rho,] H, J[, method!], args...; log=true, kwargs...)
```
Evaluates the steady state by solving iteratively the linear system <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}\hat{\rho}&space;=&space;0" title="\mathcal{L}\hat{\rho} = 0" /> while imposing a trace one condition, where the Liouvillian is defined from a Hamiltonian `H` and a vector of jump operators `J`. If `method!` is not specified, it defaults to stabilized biconjugate gradient with `l=2` GMRES steps if the operators or matrices are dense and to induced dimension reduction with a shadow space of dimension `s=8` for sparse operators or any kind of matrices (e.g. CuArrays). Further arguments are passed to the iterative solver.
```julia
Steadystate.iterative!(rho, H, J[, method!], args...; kwargs...)
rho, log = Steadystate.iterative!(rho, H, J[, method!], args...; log=true, kwargs...)
```
Same as the above but in-place.

Furthermore, adjoint jump operators `Jdagger` and decay rates `rates` can be specified as keyword arguments, as in `QuantumOptics.steadystate.master`.
