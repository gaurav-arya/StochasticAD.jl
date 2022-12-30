using StochasticAD
using Test
using Distributions
using ForwardDiff
using OffsetArrays
using ChainRulesCore
using Random

const backends = [
    StochasticAD.PrunedFIs,
    StochasticAD.PrunedFIsAggressive,
    StochasticAD.DictFIs,
]

@testset "Distributions w.r.t. continuous parameter" begin for backend in backends
    MAX = 10000
    nsamples = 100000
    rtol = 5e-2 # friendly tolerance for stochastic comparisons. TODO: more motivated choice of tolerance.
    ntests = 2 # tests for each setting

    ### Make test cases

    distributions = [
        Bernoulli,
        Geometric,
        Poisson,
        (p -> Categorical([p^2, 1 - p^2])),
        (p -> Categorical([0, p^2, 0, 0, 1 - p^2])), # check that 0's are skipped over
        (p -> Categorical([0.1, exp(p)] ./ (0.1 + exp(p)))), # test fix for #38 (floating point comparisons in Categorical logic)
        (p -> Binomial(3, p)),
        (p -> Binomial(20, p)),
    ]
    p_ranges = [
        (0.2, 0.8),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.2, 0.8),
    ]
    out_ranges = [0:1, 0:MAX, 0:MAX, 1:2, 0:5, 0:3, 0:20]
    test_cases = collect(zip(distributions, p_ranges, out_ranges))

    if backend == StochasticAD.DictFIs
        # Only test dictionary backend on Bernoulli to speed things up. Should still cover interface.
        test_cases = test_cases[1:1]
    end

    for (distr, p_range, out_range) in test_cases
        for f in [x -> x, x -> (x + 1)^2, x -> sqrt(x + 1)]
            function get_mean(p)
                dp = distr(p)
                sum(pdf(dp, i) * f(i) for i in out_range)
            end

            low_p, high_p = p_range
            for g in [p -> p, p -> high_p + low_p - p] # test both sides of derivative
                for i in 1:ntests
                    full_func = p -> f(rand(distr(g(p))))
                    p = low_p + (high_p - low_p) * rand()
                    get_deriv() = derivative_estimate(full_func, p; backend = backend)
                    triple_deriv = mean(get_deriv() for i in 1:nsamples)
                    exact_deriv = ForwardDiff.derivative(p -> get_mean(g(p)), p)
                    @test isapprox(triple_deriv, exact_deriv, rtol = rtol)
                end
            end
        end
    end
end end

@testset "Perturbing n of binomial" begin
    function get_triple_deriv(Δ)
        # Manually create a finite perturbation to avoid any randomness in its creation
        Δs = StochasticAD.similar_new(StochasticAD.PrunedFIs{Int}(), Δ, 3.5)
        st = StochasticAD.StochasticTriple{0}(5, 0, Δs)
        st_continuous = stochastic_triple(0.5)
        return derivative_contribution(rand(Binomial(st, st_continuous)))
    end
    for Δ in -2:2
        triple_deriv = mean(get_triple_deriv(Δ) for i in 1:100000)
        exact_deriv = 3.5 * 0.5 * Δ + 5
        @test isapprox(triple_deriv, exact_deriv, rtol = 5e-2)
    end
end

@testset "Nested binomials" begin
    binbin = p -> rand(Binomial(rand(Binomial(10, p)), p)) # ∼ Binomial(10, p^2)
    for p in [0.3, 0.7]
        triple_deriv = mean(derivative_estimate(binbin, p) for i in 1:100000)
        exact_deriv = 10 * 2 * p
        @test isapprox(triple_deriv, exact_deriv, rtol = 5e-2)
    end
end

@testset "Boolean comparisons" begin for backend in backends
    tested = falses(2)
    while !(all(tested))
        st = stochastic_triple(rand ∘ Bernoulli, 0.5; backend = backend)
        x = StochasticAD.value(st)
        if x == 0
            # Ensure errors on unsafe/unsupported boolean comparisons
            @test_throws Exception st>0.5
            @test_throws Exception 0.5<st
            @test_throws Exception st==0
        else
            @test st > 0.5
            @test 0.5 < st
            @test st == 1
        end
        tested[x + 1] = true
    end
    @test stochastic_triple(1.0; backend = backend) != 1
end end

@testset "Array indexing" begin
    p = 0.3
    # Test indexing into array of floats with stochastic triple index
    function array_index(p)
        arr = [3.5, 5.2, 8.4]
        index = rand(Categorical([p / 2, p / 2, 1 - p]))
        return arr[index]
    end
    array_index_mean(p) = p / 2 * 3.5 + p / 2 * 5.2 + (1 - p) * 8.4
    triple_array_index_deriv = mean(derivative_estimate(array_index, p) for i in 1:100000)
    exact_array_index_deriv = ForwardDiff.derivative(array_index_mean, p)
    @test isapprox(triple_array_index_deriv, exact_array_index_deriv, rtol = 5e-2)
    # Test indexing into array of stochastic triples with stochastic triple index
    function array_index2(p)
        arr = [3.5 * rand(Bernoulli(p)), 5.2 * rand(Bernoulli(p)), 8.4 * rand(Bernoulli(p))]
        index = rand(Categorical([p / 2, p / 2, 1 - p]))
        return arr[index]
    end
    array_index2_mean(p) = p / 2 * 3.5p + p / 2 * 5.2p + (1 - p) * 8.4p
    triple_array_index2_deriv = mean(derivative_estimate(array_index2, p) for i in 1:100000)
    exact_array_index2_deriv = ForwardDiff.derivative(array_index2_mean, p)
    @test isapprox(triple_array_index2_deriv, exact_array_index2_deriv, rtol = 5e-2)
end

@testset "Array inputs to higher level functions" begin
    # Try a deterministic test function to compare to ForwardDiff
    f(x) = (x[1] * x[2] * sin(x[3]) + exp(x[1] * x[2])) / x[3]
    x = [1, 2, π / 2]

    stochastic_ad_grad = derivative_estimate(f, x)
    stochastic_ad_grad2 = derivative_contribution.(stochastic_triple(f, x))
    fd_grad = ForwardDiff.gradient(f, x)
    @test stochastic_ad_grad ≈ fd_grad
    @test stochastic_ad_grad ≈ stochastic_ad_grad2

    # Try an OffsetArray too
    f_off(x) = (x[0] * x[1] * sin(x[2]) + exp(x[0] * x[1])) / x[2]
    x_off = OffsetArray([1, 2, π / 2], 0:2)
    stochastic_ad_grad_off = derivative_estimate(f_off, x_off)
    @test stochastic_ad_grad_off ≈ OffsetArray(stochastic_ad_grad, 0:2)

    # Test StochasticModel + stochastic_gradient combination
    m = StochasticModel(f, x)
    @test stochastic_gradient(m).p ≈ stochastic_ad_grad
end

@testset "Propagation using frule with ZeroTangent" begin
    st = stochastic_triple(0.5)

    # Verify that the rule for imag indeed gives a ZeroTangent
    value = StochasticAD.value(st)
    δ = StochasticAD.delta(st)
    @test frule((NoTangent(), δ), imag, value)[2] isa ZeroTangent
    # Test that stochastic triples flow through this rule
    out_st = imag(st)
    @test StochasticAD.value(out_st) ≈ 0
    @test StochasticAD.delta(out_st) ≈ 0
    @test isempty(out_st.Δs)
end

@testset "Unary functions converting type to fixed instance" begin for val in [0.5, 1]
    st = stochastic_triple(val)
    for op in StochasticAD.UNARY_TYPEFUNCS_WRAP
        f = getfield(Base, Symbol(op))
        out_st = f(st)
        @test out_st isa StochasticAD.StochasticTriple
        @test StochasticAD.value(out_st) ≈ f(val) ≈ f(typeof(val))
        @test StochasticAD.delta(out_st) ≈ 0
        @test isempty(out_st.Δs)
        @test f(typeof(st)) == out_st
    end
    #=
    It so happens that the UNARY_TYPEFUNCS_WRAP funcs all support both instances and types
    whereas UNARY_TYPEFUNCS_NOWRAP only supports types, so we only test types in the below,
    but this is a coincidence that may not hold in the future.
    =#
    for op in StochasticAD.UNARY_TYPEFUNCS_NOWRAP
        f = getfield(Base, Symbol(op))
        out = f(typeof(st))
        @test out isa typeof(val)
        @test out ≈ f(typeof(val))
    end
    RNG = copy(Random.GLOBAL_RNG)
    for op in StochasticAD.RNG_TYPEFUNCS_WRAP
        f = getfield(Random, Symbol(op))
        out_st = f(copy(RNG), typeof(st))
        @test out_st isa StochasticAD.StochasticTriple
        @test StochasticAD.value(out_st) ≈ f(copy(RNG), typeof(val))
        @test StochasticAD.delta(out_st) ≈ 0
        @test isempty(out_st.Δs)
    end
end end

@testset "Hashing" begin
    st = stochastic_triple(3.0)
    @test_nowarn hash(st)
    @test_nowarn hash(st, UInt(5))
    d = Dict()
    @test_nowarn d[st] = 5
    @test d[st] == 5
    @test d[3] == 5
    # Test that we get an error with discrete random dictionary indices,
    # since this isn't supported and we want to avoid silent failures.
    Δs = StochasticAD.similar_new(StochasticAD.PrunedFIs{Int}(), 1.0, 1.0)
    st = StochasticAD.StochasticTriple{0}(1.0, 0, Δs)
    @test_throws ErrorException d[rand(Bernoulli(st))]
end

@testset "Coupled comparison" begin
    Δs_1 = StochasticAD.similar_new(StochasticAD.PrunedFIs{Int}(), 1.0, 1.0)
    Δs_2 = StochasticAD.similar_new(StochasticAD.PrunedFIs{Int}(), 1.0, 1.0)
    st_1 = StochasticAD.StochasticTriple{0}(1.0, 0, Δs_1)
    st_2 = StochasticAD.StochasticTriple{0}(1.0, 0, Δs_2)
    @test st_1 == st_1
    @test_throws ErrorException st_1==st_2
end

@testset "Converting float stochastic triples to integer triples" begin
    st = stochastic_triple(0.6)
    @test round(Int, st) isa StochasticTriple
    @test StocasticAD.delta(round(Int, st)) ≈ 0
    @test round(Int, st) ≈ 1
end

@testset "Approximate comparisons" begin
    st = stochastic_triple(0.5)
    @test st ≈ st
    @test !(st ≈ st + 1)
    @test_broken stochastic_triple(Inf) ≈ stochastic_triple(Inf)
end
