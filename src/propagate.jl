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

function strip_Δs(arg)
    if arg isa StochasticTriple
        # TODO: replace check below with a more robust notion of discreteness.
        if valtype(arg) <: Integer
            return value(arg)
        else
            return ForwardDiff.Dual{tag(arg)}(value(arg), delta(arg))
        end
    else
        return arg
    end
end

"""
    propagate(f, args...; keep_deltas = Val{true})

Propagates `args` through a function `f`, handling stochastic triples appropriately.
This functionality is orthogonal to dispatch: the idea is for this function to be
the "backend" for operator overloading rules. 
Currently, we handle deterministic functions `f` with input and output supported by `Functors.jl`.
If `f` has a continuously differentiable component that should be kept,  
This function is highly experimental, and is intentionally undocumented.
"""
function propagate(f, args...; keep_deltas = Val{false})
    #= 
    TODO: maybe don't iterate through every scalar of array below, 
    but rather have special array dispatch
    =#
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
    st_rep = foldl(args_fold, args_iter)
    if !(st_rep isa StochasticTriple)
        return f(args...)
    end
    Δs_rep = st_rep.Δs

    primal_args = structural_map(get_value, args)
    input_args = keep_deltas == Val{false} ? primal_args : structural_map(strip_Δs, args)
    #= 
    TODO: the below is dangerous is general.
    It should be safe so long as f does not close over stochastic triples.
    (If f is a closure, the parameters of f should be treated like any other parameters;
    if they are stochastic triples and we are ignoring them, dangerous in general.)
    =#
    out = f(input_args...)
    val = structural_map(value, out)
    Δs_all = structural_map(Base.Fix2(get_Δs, backendtype(st_rep)), args;
        only_vals = Val{true}())
    # TODO: Coupling approach below needs to handle non-perturbable objects.
    Δs_coupled = couple(backendtype(st_rep), Δs_all; rep = Δs_rep, out_rep = val)

    function map_func(Δ_coupled)
        perturbed_args = structural_map(+, primal_args, Δ_coupled)
        #= 
        TODO: for f discrete random with randomness independent of params,
        could couple here. But difficult without a splittable RNG. 
        =#
        alt = f(perturbed_args...)
        return structural_map((x, y) -> value(x) - y, alt, val)
    end
    Δs = map(map_func, Δs_coupled; out_rep = val) # TODO: support deriv here, maybe via ForwardDiff?
    # TODO: make sure all FI backends support interface needed below
    new_out = structural_map(out, scalarize(Δs; out_rep = val)) do leaf_out, leaf_Δs
        StochasticAD.StochasticTriple{tag(st_rep)}(value(leaf_out), delta(leaf_out),
            leaf_Δs)
    end
    return new_out
end
