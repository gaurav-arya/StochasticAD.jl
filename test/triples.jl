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

@testset "Array/functor inputs to higher level functions" begin
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

    # Try a Functor
    f_func(x) = (x[1] * x[2][1] * sin(x[2][2]) + exp(x[1] * x[2][1])) / x[2][2]
    x_func = (1, [2, π / 2])
    stochastic_ad_grad_func = derivative_estimate(f_func, x_func)
    stochastic_ad_grad_func_expected = (stochastic_ad_grad[1], stochastic_ad_grad[2:3])
    compare_grad_funcs = StochasticAD.structural_map(≈, stochastic_ad_grad_func,
                                                     stochastic_ad_grad_func_expected)
    @test all(compare_grad_funcs |> StochasticAD.structural_iterate)

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
    @test round(Int, st) isa StochasticAD.StochasticTriple
    @test StochasticAD.delta(round(Int, st)) ≈ 0
    @test round(Int, st) ≈ 1
end

@testset "Approximate comparisons" begin
    st = stochastic_triple(0.5)
    @test st ≈ st
    # Check that the rtol is indeed reasonable
    @test st ≈ st + 1e-14
    @test !(st ≈ st + 1)
    @test_broken stochastic_triple(Inf) ≈ stochastic_triple(Inf)
end

@testset "Error on unmatched tags" begin
    st1 = stochastic_triple(0.5)
    st2 = stochastic_triple(x -> x^2, 0.5)
    @test_throws ArgumentError convert(typeof(st1), st2)
end

@testset "Finite perturbation backend interface" begin for FIs in backends
    #=
    Test the backend interface across the finite perturbation backends,
    which is currently a bit implicitly defined.
    =#
    V0 = Int
    V1 = Float64
    #=
    All four of the below approaches should create an empty backend,
    although the backend's internal state management may differ. 
    =#
    Δs0 = StochasticAD.similar_type(FIs, V0)() # used for first triple in computation
    Δs1 = empty(Δs0)
    Δs2 = empty(typeof(Δs0))
    Δs3 = StochasticAD.similar_empty(Δs0, V1)
    for (Δs, V) in ((Δs0, V0), (Δs1, V0), (Δs2, V0), (Δs3, V1))
        @test Δs isa FIs
        @test StochasticAD.valtype(Δs) === V
        @test isempty(Δs)
        @test iszero(derivative_contribution(Δs))
    end
    # Test creation of a single perturbation
    for Δ in (1, 3.0)
        Δs0 = StochasticAD.similar_type(FIs, V0)()
        Δs1 = StochasticAD.similar_new(Δs0, Δ, 3.0)
        @test Δs1 isa FIs
        @test StochasticAD.valtype(Δs1) === typeof(Δ)
        @test !isempty(Δs1)
        @test derivative_contribution(Δs1) == 3Δ
        # Test StochasticAD.alltrue
        @test StochasticAD.alltrue(map(_Δ -> true, Δs1))
        @test !StochasticAD.alltrue(map(_Δ -> false, Δs1))
    end
    # Test coupling
    Δ_coupleds = (3, [4.0, 5.0], (2, [3.0, 4.0]))
    for Δ_coupled in Δ_coupleds
        function get_Δs_coupled(; do_combine = false, use_get_rep = false)
            Δs0 = StochasticAD.similar_type(FIs, Int)()
            Δs1 = StochasticAD.similar_new(Δs0, 1, 3.0) # perturbation 1
            Δs2 = StochasticAD.similar_new(Δs0, 1, 2.0) # perturbation 2
            # A group of perturbations that all stem from perturbation 1. 
            Δs_all1 = StochasticAD.structural_map(Δ_coupled) do Δ
                Base.map(_Δ -> Δ, Δs1)
            end
            # A group of perturbations that all stem from perturbation 2. 
            Δs_all2 = StochasticAD.structural_map(Δ_coupled) do Δ
                Base.map(_Δ -> 2 * Δ, Δs2)
            end
            # Join them into a single structure that should be coupled
            Δs_all = (Δs_all1, Δs_all2)
            kwargs = use_get_rep ? (; rep = StochasticAD.get_rep(FIs, Δs_all)) : (;)
            if do_combine
                return StochasticAD.combine(FIs, Δs_all; kwargs...)
            else
                return StochasticAD.couple(FIs, Δs_all; kwargs...)
            end
        end
        Δs_coupled = get_Δs_coupled()
        @test StochasticAD.valtype(Δs_coupled) == typeof((Δ_coupled, Δ_coupled))
        #=
        As a test function to apply to the coupled perturbation, we apply
        a matmul followed by a sigmoid activation function and a sum.
        =#
        l = 2 * length(collect(StochasticAD.structural_iterate(Δ_coupled)))
        A = rand(l, l)
        function mapfunc(Δ_coupled)
            arr = collect(StochasticAD.structural_iterate(Δ_coupled))
            sum(x -> 1 / (1 + exp(-x)), A * arr)
        end
        # Test the above function, and also a simple sum.
        for (mapfunc, check_combine) in ((mapfunc, false),
                                         (Δ_coupled -> sum(StochasticAD.structural_iterate(Δ_coupled)),
                                          true))
            for use_get_rep in (false, true)
                function get_contribution()
                    Δs_coupled = get_Δs_coupled(; use_get_rep)
                    Δs_coupled_mapped = map(mapfunc, Δs_coupled)
                    return derivative_contribution(Δs_coupled_mapped)
                end
                zero_Δ_coupled = StochasticAD.structural_map(zero, Δ_coupled)
                expected_contribution1 = 3.0 * mapfunc((Δ_coupled, zero_Δ_coupled))
                expected_contribution2 = 2.0 * mapfunc((zero_Δ_coupled,
                                                  StochasticAD.structural_map(x -> 2x,
                                                                              Δ_coupled)))
                expected_contribution = expected_contribution1 + expected_contribution2
                @test isapprox(mean(get_contribution() for i in 1:1000),
                               expected_contribution; rtol = 5e-2)
                # For a simple sum, this should be equivalent to the combine behaviour.
                if check_combine
                    @test isapprox(mean(derivative_contribution(get_Δs_coupled(;
                                                                               do_combine = true))
                                        for i in 1:1000), expected_contribution;
                                   rtol = 5e-2)
                end
            end
        end
    end
end end

@testset "Getting information about stochastic triples" begin for backend in backends
    Random.seed!(4321)
    f(x) = rand(Bernoulli(x)) + x
    st = stochastic_triple(f, 0.5; backend)
    # Expected: 0.5 + 1.0ε + (1.0 with probability 2.0ε)
    dual = ForwardDiff.Dual(0.5, 1.0)

    @test StochasticAD.value(0.5) == 0.5
    @test StochasticAD.value(st) == 0.5
    @test StochasticAD.value(dual) == 0.5

    @test iszero(StochasticAD.delta(0.5))
    @test StochasticAD.delta(st) == 1.0
    @test StochasticAD.delta(dual) == 1.0

    #= 
    NB: since the implementation of perturbations can be backend-specific, the
    below property need not hold in general, but does for the current backends.
    =#
    @test collect(perturbations(st)) == [(1, 2.0)]

    @test StochasticAD.tag(st) === StochasticAD.Tag{typeof(f), Float64}
    @test StochasticAD.valtype(st) === Float64
    @test StochasticAD.backendtype(st) === StochasticAD.similar_type(backend, Float64)
end end
