# SteadyState.jl

[![Build Status](https://travis-ci.com/Z-Denis/SteadyState.jl.svg?token=XuYcpCDomapYmd2vHj9y&branch=master)](https://travis-ci.com/Z-Denis/SteadyState.jl)
[![Codecov](https://codecov.io/gh/Z-Denis/SteadyState.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Z-Denis/SteadyState.jl)

Tiny package for determining iteratively the steady state of time-independent Liouville superoperators thanks to [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).

#### Methods

```julia
rho = Steadystate.iterative([rho,] H, J[, method!]; args...; kwargs...)
rho, log = Steadystate.iterative([rho,] H, J[, method!]; args...; log=true, kwargs...)
```
Evaluates the steady state by solving iteratively the linear system <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}\hat{\rho}&space;=&space;0" title="\mathcal{L}\hat{\rho} = 0" />, where the Liouvillian is defined from a Hamiltonian `H` and a vector of jump operators `J`. If `method!` is not specified, it defaults to stabilized biconjugate gradient with `l=2` GMRES steps if the operators or matrices are dense and to induced dimension reduction with a shadow space of dimension `s=8` for sparse operators or any kind of matrices (e.g. CuArrays). Further arguments are passed to the iterative solver.
```julia
Steadystate.iterative!(rho, H, J[, method!]; args...; kwargs...)
rho, log = Steadystate.iterative!(rho, H, J[, method!]; args...; log=true, kwargs...)
```
Same as the above but in-place.
