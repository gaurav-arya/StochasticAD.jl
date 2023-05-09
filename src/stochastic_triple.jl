""" 
    StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}}

Stores the primal value of the computation, alongside a "dual" component
representing an infinitesimal change, and a "triple" component that tracks
discrete change(s) with infinitesimal probability. 

Pretty printed as "value + δε + ({pretty print of Δs})".

## Constructor

- `value`: the primal value.
- `δ`: the value of the almost-sure derivative, i.e. the rate of "infinitesimal" change.
- `Δs`: alternate values with associated weights, i.e. Finite perturbations with Infinitesimal probability,
        represented by a backend `FIs <: AbstractFIs`.
"""
struct StochasticTriple{T, V <: Real, FIs <: AbstractFIs{V}} <: Real
    value::V
    δ::V # infinitesimal change
    Δs::FIs # finite changes with infinitesimal probabilities # (Δ = 3, p = 1*h)
    function StochasticTriple{T, V, FIs}(value::V, δ::V,
                                         Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
        new{T, V, FIs}(value, δ, Δs)
    end
end

"""
    value(st::StochasticTriple)

Return the primal value of `st`.
"""
value(x::Real, state = nothing) = x
# Experimental method for obtaining the alternate value of a stochastic triple associated with a certain backend state.
value(st::StochasticTriple) = st.value
function value(st::StochasticTriple, state)
    st.value + filter_state(st.Δs, state)
end
#=
Support ForwardDiff.Dual for internal usage.
Assumes batch size is 1.
=#
value(d::ForwardDiff.Dual, state = nothing) = ForwardDiff.value(d)

"""
    delta(st::StochasticTriple)

Return the almost-sure derivative of `st`, i.e. the rate of infinitesimal change.
"""
delta(x::Real) = zero(x)
delta(st::StochasticTriple) = st.δ
# Support ForwardDiff.Dual for internal usage.
delta(d::ForwardDiff.Dual) = ForwardDiff.partials(d)[1]

"""
    perturbations(st::StochasticTriple)

Return the finite perturbation(s) of `st`, in a format dependent on the [backend](devdocs.md) used for storing perturbations.
"""
perturbations(x::Real) = ()
perturbations(st::StochasticTriple) = perturbations(st.Δs)

"""
    derivative_contribution(st::StochasticTriple)

Return the derivative estimate given by combining the dual and triple components of `st`.
"""
derivative_contribution(x::Real) = zero(x)
derivative_contribution(st::StochasticTriple) = st.δ + derivative_contribution(st.Δs)

"""
    tag(st::StochasticTriple)

Get the tag of a stochastic triple.
"""
tag(::StochasticTriple{T}) where {T} = T

"""
    valtype(st::StochasticTriple)

Get the underlying type of the value tracked by a stochastic triple.
"""
valtype(::StochasticTriple{T, V}) where {T, V} = V

"""
    backendtype(st::StochasticTriple)

Get the backend type of a stochastic triple.
"""
backendtype(::StochasticTriple{T, V, FIs}) where {T, V, FIs} = FIs

"""
    smooth_triple(st::StochasticTriple)

Smooth the dual and triple components of a stochastic triple into a single dual component.
Useful for avoiding unnecessary pruning when running multilinear functions on triples.
"""
smooth_triple(x::Real) = x
function smooth_triple(st::StochasticTriple{T, V, FIs}) where {T, V, FIs}
    return StochasticTriple{T}(value(st), derivative_contribution(st), empty(FIs))
end

### Extra constructors of stochastic triples

function StochasticTriple{T}(value::V, δ::V, Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
    StochasticTriple{T, V, FIs}(value, δ, Δs)
end

function StochasticTriple{T}(value::V, Δs::FIs) where {T, V, FIs <: AbstractFIs{V}}
    StochasticTriple{T}(value, zero(value), Δs)
end

function StochasticTriple{T}(value::A, δ::B,
                             Δs::FIs) where {T, A, B, C, FIs <: AbstractFIs{C}}
    V = promote_type(A, B, C)
    StochasticTriple{T}(convert(V, value), convert(V, δ), similar_type(FIs, V)(Δs))
end

### Conversion rules

# TODO: is this the right thing to do? Maybe, different from the promote case because there V was guaranteed to be an ancestor. 
# Also, bad to do when already same type?
function Base.convert(::Type{StochasticTriple{T1, V, FIs}},
                      x::StochasticTriple{T2}) where {T1, T2, V, FIs}
    (T1 !== T2) && throw(ArgumentError("Tags of combined stochastic triples do not match."))
    StochasticTriple{T1, V, FIs}(convert(V, x.value), convert(V, x.δ), FIs(x.Δs))
end

# TODO: ForwardDiff's promotion rules are a little more complicated, see https://github.com/JuliaDiff/ForwardDiff.jl/issues/322
# May need to look into why and possibly use them here too.
function Base.promote_rule(::Type{StochasticTriple{T, V1, FIs}},
                           ::Type{StochasticTriple{T, V2, FIs2}}) where {T, V1, FIs, V2,
                                                                         FIs2}
    V = promote_type(V1, V2)
    StochasticTriple{T, V, similar_type(FIs, V)}
end

function Base.promote_rule(::Type{StochasticTriple{T, V1, FIs}},
                           ::Type{V2}) where {T, V1, FIs, V2 <: Real}
    V = promote_type(V1, V2)
    StochasticTriple{T, V, similar_type(FIs, V)}
end

function Base.convert(::Type{StochasticTriple{T, V, FIs}}, x::Real) where {T, V, FIs}
    StochasticTriple{T, V, FIs}(convert(V, x), zero(V), empty(FIs))
end

### Creating the first stochastic triple in a computation

function StochasticTriple{T}(value::V, δ::V, backend::AbstractFIsBackend) where {T, V}
    StochasticTriple{T}(value, δ, create_Δs(backend, V))
end

function StochasticTriple{T}(value::V, backend::AbstractFIsBackend) where {T, V}
    StochasticTriple{T}(value, zero(V), backend)
end

function StochasticTriple{T}(value::A, δ::B, backend::AbstractFIsBackend) where {T, A, B}
    V = promote_type(A, B)
    StochasticTriple{T}(convert(V, value), convert(V, δ), backend)
end

### Showing a stochastic triple

function Base.summary(::StochasticTriple{T, V}) where {T, V}
    return "StochasticTriple of $V"
end

function Base.show(io::IO, ::MIME"text/plain", st::StochasticTriple)
    println(io, "$(summary(st)):")
    show(io, st)
end

function Base.show(io::IO, st::StochasticTriple)
    print(io, "$(st.value) + $(st.δ)ε")
    if (!isempty(st.Δs))
        print(io, " + ($(repr(st.Δs)))")
    end
end

### Higher level functions

struct Tag{F, V}
end

function stochastic_triple_direction(f, p::V, direction;
                                     backend = PrunedFIsBackend()) where {V}
    Δs = create_Δs(backend, Int) # TODO: necessity of hardcoding some type here suggests interface improvements
    sts = structural_map(p, direction) do p_i, direction_i
        StochasticTriple{Tag{typeof(f), V}}(p_i, direction_i,
                                            similar_empty(Δs, typeof(p_i)))
    end
    return f(sts)
end

"""
    stochastic_triple(X, p; backend=PrunedFIsBackend(), direction=nothing)
    stochastic_triple(p; backend=PrunedFIsBackend(), direction=nothing)

For any `p` that is supported by [`Functors.jl`](https://fluxml.ai/Functors.jl/stable/),
e.g. scalars or abstract arrays,
differentiate the output with respect to each value of `p`,
returning an output of similar structure to `p`, where a particular value contains
the stochastic-triple output of `X` when perturbing the corresponding value in `p`
(i.e. replacing the original value `x` with `x + ε`).

When `direction` is provided, return only the stochastic-triple output of `X` with respect to a perturbation
of `p` in that particular direction.
When `X` is not provided, the identity function is used. 

The `backend` keyword argument describes the algorithm used by the third component
of the stochastic triple, see [technical details](devdocs.md) for more details.

# Example
```jldoctest
julia> using Distributions, Random, StochasticAD; Random.seed!(4321);

julia> stochastic_triple(rand ∘ Bernoulli, 0.5)
StochasticTriple of Int64:
0 + 0ε + (1 with probability 2.0ε, tag 1)
```
"""
function stochastic_triple(f, p; direction = nothing, kwargs...)
    if direction !== nothing
        return stochastic_triple_direction(f, p, direction; kwargs...)
    end
    counter = begin
        c = 0
        (_) -> begin
            c += 1
            return c
        end
    end
    indices = structural_map(counter, p)
    map_func = perturbed_index -> begin
        direction = structural_map(indices, p) do i, p_i
            i == perturbed_index ? one(p_i) : zero(p_i)
        end
        stochastic_triple_direction(f, p, direction; kwargs...)
    end
    return structural_map(map_func, indices)
end

stochastic_triple(p; kwargs...) = stochastic_triple(identity, p; kwargs...)

@doc raw"""
    derivative_estimate(X, p; backend=PrunedFIsBackend(), direction=nothing)

Compute an unbiased estimate of ``\frac{\mathrm{d}\mathbb{E}[X(p)]}{\mathrm{d}p}``, 
the derivative of the expectation of the random function `X(p)` with respect to its input `p`. 

Both `p` and `X(p)` can be any object supported by [`Functors.jl`](https://fluxml.ai/Functors.jl/stable/),
e.g. scalars or abstract arrays. 
The output of `derivative_estimate` has the same outer structure as `p`, but with each
scalar in `p` replaced by a derivative estimate of `X(p)` with respect to that entry.

For example, if `X(p) <: AbstractMatrix` and `p <: Real`, then the output would be a matrix.
The `backend` keyword argument describes the algorithm used by the third component
of the stochastic triple, see [technical details](devdocs.md) for more details.

When `direction` is provided, the output is only differentiated with respect to a perturbation
of `p` in that direction.

!!! note 
    Since `derivative_estimate` performs forward-mode AD, the required computation time scales
    linearly with the number of parameters in `p` (but is unaffected by the number of parameters in `X(p)`).
# Example
```jldoctest
julia> using Distributions, Random, StochasticAD; Random.seed!(4321);

julia> derivative_estimate(rand ∘ Bernoulli, 0.5) # A random quantity that averages to the true derivative.
2.0

julia> derivative_estimate(x -> [rand(Bernoulli(x * i/4)) for i in 1:3], 0.5)
3-element Vector{Float64}:
 0.2857142857142857
 0.6666666666666666
 0.0
```
"""
function derivative_estimate(f, p; kwargs...)
    StochasticAD.structural_map(derivative_contribution, stochastic_triple(f, p; kwargs...))
end
