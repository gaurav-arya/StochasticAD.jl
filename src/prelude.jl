const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

const UNARY_PREDICATES = [isinf, isnan, isfinite, iseven, isodd, isreal, isinteger]

const BINARY_PREDICATES = [
    isequal,
    isless,
    <,
    >,
    ==,
    !=,
    <=,
    >=
]

const UNARY_TYPEFUNCS_NOWRAP = [Base.rtoldefault]
const UNARY_TYPEFUNCS_WRAP = [
    typemin,
    typemax,
    floatmin,
    floatmax,
    zero,
    one
]
const RNG_TYPEFUNCS_WRAP = [rand, randn, randexp]

"""
    structural_iterate(args)

Internal helper function for iterating through the scalar values of a functor, 
where AbstractFIs are also counted as scalars.
"""
function structural_iterate(args)
    make_iterator(x) = x isa AbstractArray ? x : (x,)
    exclude(x) = Functors.isleaf(x) || (x isa AbstractFIs)
    iter = fmap(make_iterator, args; walk = Functors.IterateWalk(), cache = nothing,
        exclude)
    return iter
end
structural_iterate(args::NTuple{N,Union{Real, AbstractFIs}}) where {N} = args
structural_iterate(args::AbstractArray{T}) where {T<:Union{Real,AbstractFIs}} = args
structural_iterate(args::T) where {T<:Real} = (args,)

"""
    structural_map(f, args)

Internal helper function for a structure-preserving map, 
often to be used on a function's input/output arguments. 
Currently uses [fmap](https://fluxml.ai/Functors.jl/stable/api/#Functors.fmap) 
from Functors.jl as a backend.
"""
function structural_map(f, args...; only_vals = nothing)
    walk = if only_vals isa Val{true}
        Functors.StructuralWalk()
    elseif (only_vals isa Val{false}) || isnothing(only_vals)
            Functors.DefaultWalk()
    else
        error("Unsupported argument only_vals = $only_vals")
    end
    fmap((args...) -> args[1] isa AbstractArray ? f.(args...) : f(args...), args...;
        cache = nothing,
        walk)
end
