# SteadyState.jl

[![Build Status](https://travis-ci.com/Z-Denis/SteadyState.jl.svg?token=XuYcpCDomapYmd2vHj9y&branch=master)](https://travis-ci.com/Z-Denis/SteadyState.jl)
[![Codecov](https://codecov.io/gh/Z-Denis/SteadyState.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Z-Denis/SteadyState.jl)

Tiny package for determining iteratively the steady state of time-independent Liouville superoperators thanks to [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) and [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl).

#### Methods

```julia
rho = steadystate_bicg(H, J, l; log=false, kwargs...)
rho, log = steadystate_bicg(H, J, l; log=true, kwargs...)
```
Evaluates the steady state by solving iteratively the linear system <img src="https://latex.codecogs.com/gif.latex?\mathcal{L}\hat{\rho}&space;=&space;0" title="\mathcal{L}\hat{\rho} = 0" /> via the stabilized biconjugate gradient method with `l` GMRES steps. The Hamiltonian `H` and the jump operators `J` are to be dense. Sparse operators can be handled, provided one gets rid of BLAS in [this line](https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/src/bicgstabl.jl#L128) (e.g. by changing it to `@inbounds @views it.x .+= it.rs[:,1:it.l] * it.γ`  in some local branch of [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl)). Keyword arguments are passed to the iterative solver.
```julia
steadystate_bicg!(rho, H, J, l; log=false, kwargs...)
```
Same as the above with an initial condition.
```julia
steadystate_iterative!(rho, H, J, method!, args...; kwargs...)
```
Same as the above but accepting any quantum operator or array and any inplace method (e.g. `gmres!`, for dense operators, or `idrs!`, compatible with sparse operators as well, from [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl)). `args` and `kwargs` are passed on to the iterative solver.

#### To do
Add methods for determining the steady state of Bloch-Redfield master equations.
