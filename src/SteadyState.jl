module SteadyState

include("bicgstab.jl")
export steadystate_bicg, steadystate_bicg!, steadystate_iterative!
#include("indexing.jl")

end # module
