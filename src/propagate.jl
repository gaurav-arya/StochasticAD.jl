"""
A version of `value`` that allows unrecognized args to pass through. 
"""
function get_value(arg)
    if arg isa StochasticTriple
        return value(arg)
    else
        # potentially dangerous, see also note in get_Δs
        return arg
    end
end

function get_Δs(arg, FIs)
    if arg isa StochasticTriple
        return arg.Δs
    else
        #=
        this case is a bit dangerous: perturbations could be dropped here
        if a leaf of a functor somehow contains a type that is not one of 
        the two above.
        =#
        return empty(similar_type(FIs, typeof(arg)))
    end
end

function strip_Δs(arg; use_dual = Val(true))
    if arg isa StochasticTriple
        # TODO: replace check below with a more robust notion of discreteness.
        if valtype(arg) <: Integer
            return value(arg)
        else
            if use_dual isa Val{true}
                return ForwardDiff.Dual{tag(arg)}(value(arg), delta(arg))
            else
                return StochasticAD.StochasticTriple{tag(arg)}(
                    value(arg), delta(arg), empty(arg.Δs))
            end
        end
    else
        return arg
    end
end

# Stochastic triples hidden in `f` (captured variables of a closure, or fields of a callable
# struct) are not coupled below: `f` is re-run on the alternatives with them left unperturbed,
# so their perturbations would be silently dropped. Arrays, tuples, boxes and nested structs
# are searched up to a small depth; element types that cannot hold a triple are skipped.
_may_hold_triple(::Type{<:StochasticTriple}) = true
_may_hold_triple(::Type{<:Union{Number, AbstractString, Symbol, Char}}) = false
function _may_hold_triple(T::Type)
    isconcretetype(T) || return true
    isbitstype(T) && return any(_may_hold_triple, fieldtypes(T))
    return true
end

function _captures_triple(x, depth = 0)
    x isa StochasticTriple && return true
    (depth > 4 || !_may_hold_triple(typeof(x))) && return false
    if x isa Core.Box
        return isdefined(x, :contents) && _captures_triple(x.contents, depth + 1)
    elseif x isa AbstractArray
        _may_hold_triple(eltype(x)) || return false
        return any(y -> _captures_triple(y, depth + 1), x)
    elseif x isa Union{Tuple, NamedTuple}
        return any(y -> _captures_triple(y, depth + 1), x)
    elseif x isa Union{Module, Type, DataType}
        return false
    end
    return any(i -> isdefined(x, i) && _captures_triple(getfield(x, i), depth + 1),
        1:nfields(x))
end

"""
    propagate(f, args...; keep_deltas = Val(false))

Propagates `args` through a function `f`, handling stochastic triples by independently running `f` on the primal
and the alternatives, rather than by inspecting the internals of `f` (which may possibly be unsupported by `StochasticAD`).
Currently handles deterministic functions `f` with any input and output that is `fmap`-able by `Functors.jl`.
If `f` has a continuously differentiable component, provide `keep_deltas = Val(true)`.
Stochastic triples must reach `f` through `args`: if `f` captures one (e.g. a closure over a
stochastic triple), an `ArgumentError` is thrown, since its perturbations could not be propagated.

This functionality is orthogonal to dispatch: the idea is for this function to be the "backend" for operator 
overloading rules based on dispatch. For example:

```jldoctest
using StochasticAD, Distributions
import Random # hide
Random.seed!(4321) # hide

function mybranch(x)
    str = repr(x) # string-valued intermediate!
    if length(str) < 2
        return 3
    else
        return 7
    end
end

function f(x)
    return mybranch(9 + rand(Bernoulli(x)))
end

# stochastic_triple(f, 0.5) # this would fail

# Add a dispatch rule for mybranch using StochasticAD.propagate
mybranch(x::StochasticAD.StochasticTriple) = StochasticAD.propagate(mybranch, x)

stochastic_triple(f, 0.5) # now works

# output

StochasticTriple of Int64:
3 + 0ε + (4 with probability 2.0ε)
```

!!! warning
    This function is experimental and subject to change.
"""
function propagate(f,
        args...;
        keep_deltas = Val(false),
        provided_st_rep = nothing,
        deriv = nothing)
    # TODO: support kwargs to f (or just use kwfunc in macro)
    #= 
    TODO: maybe don't iterate through every scalar of array below, 
    but rather have special array dispatch
    =#
    st_rep = if provided_st_rep === nothing
        args_iter = structural_iterate(args)
        function args_fold(arg1, arg2)
            if arg1 isa StochasticTriple
                if (arg2 isa StochasticTriple) && (tag(arg1) !== tag(arg2))
                    throw(ArgumentError("Tags of combined stochastic triples do not match!"))
                end
                return arg1
            else
                return arg2
            end
        end
        foldl(args_fold, args_iter)
    else
        provided_st_rep
    end

    if !(st_rep isa StochasticTriple)
        return f(args...)
    end
    _captures_triple(f) && throw(ArgumentError("`f` captures a stochastic triple, whose " *
                        "perturbations `propagate` cannot track. Pass it to `propagate` as an argument instead."))

    primal_args = structural_map(get_value, args)
    input_args = keep_deltas isa Val{false} ? primal_args : structural_map(strip_Δs, args)
    # f may not close over stochastic triples (checked above), so only `args` are perturbed.
    out = f(input_args...)
    val = structural_map(value, out)
    # TODO: what does the only_vals do in the below and why?
    Δs_all = structural_map(Base.Fix2(get_Δs, backendtype(st_rep)), args;
        only_vals = Val{true}())
    # TODO: Coupling approach below needs to handle non-perturbable objects.
    Δs_coupled = couple(backendtype(st_rep), Δs_all; rep = st_rep.Δs, out_rep = val)

    function map_func(Δ_coupled)
        perturbed_args = structural_map(+, primal_args, Δ_coupled)
        #= 
        TODO: for f discrete random with randomness independent of params,
        could couple here. But difficult without a splittable RNG. 
        =#
        alt = f(perturbed_args...)
        return structural_map((x, y) -> value(x) - y, alt, val)
    end
    Δs = map(map_func, Δs_coupled; out_rep = val, deriv)
    # TODO: make sure all FI backends support interface needed below
    new_out = structural_map(out, scalarize(Δs; out_rep = val)) do leaf_out, leaf_Δs
        StochasticAD.StochasticTriple{tag(st_rep)}(value(leaf_out), delta(leaf_out),
            leaf_Δs)
    end
    return new_out
end
