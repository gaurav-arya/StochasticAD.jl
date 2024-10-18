using StochasticAD
using Test
using Distributions
using ForwardDiff
using OffsetArrays
using ChainRulesCore
using Random
using Zygote

const backends = [
    PrunedFIsBackend(),
    PrunedFIsAggressiveBackend(),
    DictFIsBackend()
]

const backends_smoothed = [
    SmoothedFIsBackend(),
    StrategyWrapperFIsBackend(SmoothedFIsBackend(), StochasticAD.TwoSidedStrategy())
]

@testset "Distributions w.r.t. continuous parameter" begin
    for backend in vcat(backends,
        backends_smoothed,
        :smoothing_autodiff)
        MAX = 10000
        nsamples = 100000
        rtol = 5e-2 # friendly tolerance for stochastic comparisons. TODO: more motivated choice of tolerance.

        ### Make test cases

        distributions = [
            Bernoulli,
            Geometric,
            Poisson,
            (p -> Categorical([p^2, 1 - p^2])),
            (p -> Categorical([0, p^2, 0, 0, 1 - p^2])), # check that 0's are skipped over
            (p -> Categorical([1.0, exp(p)] ./ (1.0 + exp(p)))), # test fix for #38 (floating point comparisons in Categorical logic)
            (p -> Binomial(3, p)),
            (p -> Binomial(20, p))
        ]
        p_ranges = [(0.2, 0.8) for _ in 1:8]
        out_ranges = [0:1, 0:MAX, 0:MAX, 1:2, 1:5, 1:2, 0:3, 0:20]
        test_cases = collect(zip(distributions, p_ranges, out_ranges))
        test_funcs = [x -> 7 * x - 3, x -> (x + 1)^2, x -> sqrt(x + 1)]

        if backend isa DictFIsBackend
            # Only test dictionary backend on Bernoulli to speed things up. Should still cover interface.
            test_cases = test_cases[1:1]
        elseif backend == :smoothing_autodiff || backend in backends_smoothed
            # Only test smoothing backend on each unique distribution once to seed tests up. 
            test_cases = vcat(test_cases[1:4], test_cases[7])
            # Only test unbiasedness of smoothing for linear function
            test_funcs = test_funcs[1:1]
        end

        for (distr, p_range, out_range) in test_cases
            for f in test_funcs
                function get_mean(p)
                    dp = distr(p)
                    sum(pdf(dp, i) * f(i) for i in out_range)
                end

                low_p, high_p = p_range
                for g in [p -> p, p -> high_p + low_p - p] # test both sides of derivative
                    full_func = f ∘ rand ∘ distr ∘ g
                    p = low_p + (high_p - low_p) * rand()
                    exact_deriv = ForwardDiff.derivative(p -> get_mean(g(p)), p)
                    if backend == :smoothing_autodiff
                        batched_full_func(p) = mean([full_func(p) for i in 1:nsamples])
                        # The array input used for ForwardDiff below is a trick to test multiple partials
                        triple_deriv_forward = mean(ForwardDiff.gradient(
                            arr -> batched_full_func(sum(arr)),
                            [2 * p, -p]))
                        triple_deriv_backward = Zygote.gradient(batched_full_func, p)[1]
                        @test isapprox(triple_deriv_forward, exact_deriv, rtol = rtol)
                        @test isapprox(triple_deriv_backward, exact_deriv, rtol = rtol)
                    else
                        get_deriv = () -> derivative_estimate(full_func, p; backend)
                        triple_deriv = mean(get_deriv() for i in 1:nsamples)
                        @test isapprox(triple_deriv, exact_deriv, rtol = rtol)
                    end
                end
            end
        end
    end
end

@testset "Perturbing n of binomial" begin
    function get_triple_deriv(Δ)
        # Manually create a finite perturbation to avoid any randomness in its creation
        Δs = StochasticAD.similar_new(StochasticAD.create_Δs(PrunedFIsBackend(), Int), Δ,
            3.5)
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

@testset "Boolean comparisons" begin
    for backend in backends
        tested = falses(2)
        while !(all(tested))
            st = stochastic_triple(rand ∘ Bernoulli, 0.5; backend)
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
        @test stochastic_triple(1.0; backend) != 1
    end
end

@testset "Array indexing" begin
    for backend in vcat(backends, backends_smoothed)
        p = 0.3
        # Test indexing into array of floats with stochastic triple index
        arr = [3.5, 5.2, 8.4]
        (backend in backends_smoothed) && (arr[3] = 6.9) # make linear for smoothing test
        function array_index(p)
            index = rand(Categorical([p / 2, p / 2, 1 - p]))
            return arr[index]
        end
        array_index_mean(p) = sum([p / 2, p / 2, (1 - p)] .* arr)
        triple_array_index_deriv = mean(derivative_estimate(array_index, p; backend)
        for i in 1:50000)
        exact_array_index_deriv = ForwardDiff.derivative(array_index_mean, p)
        @test isapprox(triple_array_index_deriv, exact_array_index_deriv, rtol = 5e-2)
        # Don't run subsequent tests with smoothing backend
        (backend in backends_smoothed) && continue
        # Test indexing into array of stochastic triples with stochastic triple index
        function array_index2(p)
            arr2 = [rand(Bernoulli(p)), rand(Bernoulli(p)), rand(Bernoulli(p))] .* arr
            index = rand(Categorical([p / 2, p / 2, 1 - p]))
            return arr2[index]
        end
        array_index2_mean(p) = sum([p / 2 * p, p / 2 * p, (1 - p) * p] .* arr)
        triple_array_index2_deriv = mean(derivative_estimate(array_index2, p; backend)
        for i in 1:50000)
        exact_array_index2_deriv = ForwardDiff.derivative(array_index2_mean, p)
        @test isapprox(triple_array_index2_deriv, exact_array_index2_deriv, rtol = 5e-2)
        # Test case where triple and alternate array value are coupled
        function array_index3(p)
            st = rand(Bernoulli(p))
            arr2 = [-5, st]
            return arr2[st + 1]
        end
        array_index3_mean(p) = -5 * (1 - p) + 1 * p
        triple_array_index3_deriv = mean(derivative_estimate(array_index3, p; backend)
        for i in 1:50000)
        exact_array_index3_deriv = ForwardDiff.derivative(array_index3_mean, p)
        @test isapprox(triple_array_index3_deriv, exact_array_index3_deriv, rtol = 5e-2)
    end
end

@testset "Array/functor inputs to higher level functions" begin
    for backend in backends
        # Try a deterministic test function to compare to ForwardDiff
        f(x) = (x[1] * x[2] * sin(x[3]) + exp(x[1] * x[2])) / x[3]
        x = [1, 2, π / 2]

        stochastic_ad_grad = derivative_estimate(f, x; backend)
        stochastic_ad_grad2 = derivative_contribution.(stochastic_triple(f, x; backend))
        stochastic_ad_grad_firsttwo = derivative_estimate(
            f, x; direction = [1.0, 1.0, 0.0],
            backend)
        fd_grad = ForwardDiff.gradient(f, x)
        @test stochastic_ad_grad ≈ fd_grad
        @test stochastic_ad_grad ≈ stochastic_ad_grad2
        @test stochastic_ad_grad[1] + stochastic_ad_grad[2] ≈ stochastic_ad_grad_firsttwo

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

@testset "Unary functions converting type to fixed instance" begin
    for val in [0.5, 1]
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
    end
end

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
    Δs = StochasticAD.similar_new(StochasticAD.create_Δs(PrunedFIsBackend(), Int), 1.0, 1.0)
    st = StochasticAD.StochasticTriple{0}(1.0, 0, Δs)
    @test_throws ErrorException d[rand(Bernoulli(st))]
end

@testset "Coupled comparison" begin
    Δs_1 = StochasticAD.similar_new(StochasticAD.create_Δs(PrunedFIsBackend(), Int), 1.0,
        1.0)
    Δs_2 = StochasticAD.similar_new(StochasticAD.create_Δs(PrunedFIsBackend(), Int), 1.0,
        1.0)
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

@testset "Finite perturbation backend interface" begin
    for backend in vcat(backends,
        backends_smoothed)
        # this boolean may need to become more fine-grained in the future
        is_smoothed_backend = backend in backends_smoothed
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
        Δs0 = StochasticAD.create_Δs(backend, V0) # used to create first triple in computation
        FIs = typeof(Δs0)
        Δs1 = empty(Δs0)
        Δs2 = empty(typeof(Δs0))
        Δs3 = StochasticAD.similar_empty(Δs0, V1)
        for (Δs, V) in ((Δs0, V0), (Δs1, V0), (Δs2, V0), (Δs3, V1))
            @test StochasticAD.valtype(Δs) === V
            @test Δs isa StochasticAD.similar_type(FIs, V)
            !is_smoothed_backend && @test isempty(Δs)
            @test iszero(derivative_contribution(Δs))
        end
        # Test creation of a single perturbation
        for Δ in (1, 3.0)
            Δs0 = StochasticAD.create_Δs(backend, V0)
            Δs1 = StochasticAD.similar_new(Δs0, Δ, 3.0)
            @test StochasticAD.valtype(Δs1) === typeof(Δ)
            @test Δs1 isa StochasticAD.similar_type(FIs, typeof(Δ))
            !is_smoothed_backend && @test !isempty(Δs1)
            @test derivative_contribution(Δs1) == 3Δ
            # Test StochasticAD.alltrue
            @test StochasticAD.alltrue(_Δ -> true, Δs1)
            @test !StochasticAD.alltrue(_Δ -> false, Δs1) || is_smoothed_backend
            # Test map
            # We use a dummy deriv here and below. TODO: use a more interesting dummy for better testing.
            Δs1_map = Base.map(Δ -> Δ^2, Δs1; deriv = identity, out_rep = Δ)
            !is_smoothed_backend && @test derivative_contribution(Δs1_map) ≈ Δ^2 * 3.0
            # Test map with weight (make a new copy so that original does not get reweighted)
            Δs2 = StochasticAD.similar_new(StochasticAD.create_Δs(backend, V0), Δ, 3.0)
            Δs2_weight_map = StochasticAD.weighted_map_Δs((Δ, _) -> (Δ^2, 2.0),
                Δs2;
                deriv = identity,
                out_rep = Δ)
            !is_smoothed_backend &&
                @test derivative_contribution(Δs2_weight_map) ≈ Δ^2 * 3.0 * 2.0
            # Also test scale
            w2 = derivative_contribution(Δs2)
            Δs2_scaled = StochasticAD.scale(Δs2, 2.0)
            w2_scaled = derivative_contribution(Δs2_scaled)
            @test w2_scaled ≈ 2.0 * w2
            # Test map_Δs with filter state
            if !is_smoothed_backend
                Δs1_plus_Δs0 = StochasticAD.map_Δs(
                    (Δ, state) -> Δ +
                                  StochasticAD.filter_state(Δs0,
                        state),
                    Δs1)
                @test derivative_contribution(Δs1_plus_Δs0) ≈ Δ * 3.0
                Δs1_plus_mapped = StochasticAD.map_Δs(
                    (Δ, state) -> Δ +
                                  StochasticAD.filter_state(Δs1,
                        state),
                    Δs1_map)
                @test derivative_contribution(Δs1_plus_mapped) ≈ Δ * 3.0 + Δ^2 * 3.0
            end
        end
        # Test coupling
        Δ_coupleds = (3, [4.0, 5.0], (2, [3.0, 4.0]))
        for Δ_coupled in Δ_coupleds
            function get_Δs_coupled(; do_combine = false, use_get_rep = false)
                Δs0 = StochasticAD.create_Δs(backend, Int)
                Δs1 = StochasticAD.similar_new(Δs0, 1, 3.0) # perturbation 1
                Δs2 = StochasticAD.similar_new(Δs0, 1, 2.0) # perturbation 2
                # A group of perturbations that all stem from perturbation 1. 
                Δs_all1 = StochasticAD.structural_map(Δ_coupled) do Δ
                    Base.map(_Δ -> Δ, Δs1; deriv = identity, out_rep = Δ)
                end
                # A group of perturbations that all stem from perturbation 2. 
                Δs_all2 = StochasticAD.structural_map(Δ_coupled) do Δ
                    Base.map(_Δ -> 2 * Δ, Δs2; deriv = (δ -> 2δ), out_rep = Δ)
                end
                # Join them into a single structure that should be coupled
                Δs_all = (Δs_all1, Δs_all2)
                kwargs = use_get_rep ? (; rep = StochasticAD.get_rep(FIs, Δs_all)) : (;)
                if do_combine
                    return StochasticAD.combine(FIs, Δs_all; kwargs...)
                else
                    return StochasticAD.couple(FIs, Δs_all;
                        out_rep = (Δ_coupled, Δ_coupled),
                        kwargs...)
                end
            end
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
            for use_get_rep in (false, true)
                Δs_coupled = get_Δs_coupled(; use_get_rep)
                @test StochasticAD.valtype(Δs_coupled) == typeof((Δ_coupled, Δ_coupled))
                for (mapfunc, check_combine) in ((mapfunc, false),
                    (Δ_coupled -> sum(StochasticAD.structural_iterate(Δ_coupled)),
                        true))
                    function get_contribution()
                        Δs_coupled = get_Δs_coupled(; use_get_rep)
                        Δs_coupled_mapped = map(mapfunc, Δs_coupled; deriv = (δ -> 1.0),
                            out_rep = 0.0)
                        return derivative_contribution(Δs_coupled_mapped)
                    end
                    zero_Δ_coupled = StochasticAD.structural_map(zero, Δ_coupled)
                    expected_contribution1 = 3.0 * mapfunc((Δ_coupled, zero_Δ_coupled))
                    expected_contribution2 = 2.0 * mapfunc((zero_Δ_coupled,
                        StochasticAD.structural_map(x -> 2x,
                            Δ_coupled)))
                    expected_contribution = expected_contribution1 + expected_contribution2
                    if !is_smoothed_backend
                        @test isapprox(mean(get_contribution() for i in 1:1000),
                            expected_contribution; rtol = 5e-2)
                    end
                    # For a simple sum, this should be equivalent to the combine behaviour.
                    if check_combine && !is_smoothed_backend
                        @test isapprox(
                            mean(derivative_contribution(get_Δs_coupled(;
                                     do_combine = true))
                            for i in 1:1000),
                            expected_contribution;
                            rtol = 5e-2)
                    end
                    # Check scalarize
                    Δs_coupled2 = StochasticAD.couple(FIs,
                        StochasticAD.scalarize(Δs_coupled;
                            out_rep = (Δ_coupled,
                                Δ_coupled)),
                        out_rep = (Δ_coupled, Δ_coupled))
                    @test derivative_contribution(map(mapfunc, Δs_coupled;
                        deriv = (δ -> 1.0),
                        out_rep = 0.0)) ≈
                          derivative_contribution(map(mapfunc, Δs_coupled2;
                        deriv = (δ -> 1.0),
                        out_rep = 0.0))
                end
            end
        end
    end
end

@testset "Getting information about stochastic triples" begin
    for backend in vcat(backends,
        backends_smoothed)
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

        if !(backend in backends_smoothed)
            #= 
            NB: since the implementation of perturbations can be backend-specific, the
            below property need not hold in general, but does for the current non-smoothed backends.
            =#
            p = only(perturbations(st))
            @test p.Δ == 1 && p.weight == 2.0
            @test derivative_contribution(st) == 3.0
        else
            # Since smoothed algorithm uses the two-sided strategy, we get a different derivative contribution.
            @test derivative_contribution(st) == 2.0
        end

        @test StochasticAD.tag(st) === StochasticAD.Tag{typeof(f), Float64}
        @test StochasticAD.valtype(st) === Float64
        @test StochasticAD.valtype(st.Δs) === Float64
    end
end

@testset "Propagation via StochasticAD.propagate" begin
    for backend in backends
        function form_triple(primal, δ, Δ, Δs_base)
            Δs = map(_Δ -> Δ, Δs_base)
            return StochasticAD.StochasticTriple{0}(primal, δ, Δs)
        end

        function test_propagate(f, primals, Δs; test_deltas = false)
            Δs_base = StochasticAD.similar_new(StochasticAD.create_Δs(backend, Int),
                0, 1.0)
            _form_triple(x, δ, Δ) = form_triple(x, δ, Δ, Δs_base)
            out = f(primals...)
            out_Δ_expected = StochasticAD.structural_map(-,
                f(StochasticAD.structural_map(+,
                    primals,
                    Δs)...),
                f(primals...))
            if test_deltas
                duals = StochasticAD.structural_map(primals) do x
                    x isa AbstractFloat ? ForwardDiff.Dual{0}(x, rand(typeof(x))) : x
                end
                δs = StochasticAD.structural_map(StochasticAD.delta, duals)
                out_δ_expected = StochasticAD.structural_map(StochasticAD.delta,
                    f(duals...))
            else
                δs = StochasticAD.structural_map(zero, primals)
                out_δ_expected = StochasticAD.structural_map(zero, out)
            end
            input_sts = StochasticAD.structural_map(_form_triple, primals, δs, Δs)
            out_st = StochasticAD.propagate(f, input_sts...; keep_deltas = Val{test_deltas})
            # Test type
            StochasticAD.structural_map(out_st, out, out_δ_expected,
                out_Δ_expected) do x_st, x, δ, Δ
                @test x_st isa StochasticAD.StochasticTriple{0, typeof(x)}
                @test StochasticAD.value(x_st) == x
                @test StochasticAD.delta(x_st) ≈ δ
                p = only(perturbations(x_st))
                @test p.Δ == Δ && p.weight == 1.0
            end
        end

        #=
        Test propagation through some simple functions. 
            f1: a simple if statement.
            f2: involves array-containing-fucntor input and output.
            f3: involves array-containing-functor input, but real output.
            f4: length ∘ repr (real or array input, real output).
            f5: mutates input array! Broken since unsupported.
            f6: the first-arg (blob) should just be passed through without attempting
                to perturb. Broken since unsupported.
            f7: involves matrix-containing-functor input and output.
        =#
        function f1(x)
            if x == 0
                return 1
            elseif x == 3
                return 2
            else
                return 5
            end
        end

        @test StochasticAD.propagate(f1, 0) === f1(0)
        for (primal, Δ) in [(0, 3), (0, 4), (3, -1)]
            test_propagate(f1, (primal,), (Δ,))
        end

        function f2(arr, scalar)
            if sum(arr) + scalar <= 5
                return arr .* scalar, sum(arr) * scalar
            else
                return arr .- scalar, sum(arr) - scalar
            end
        end
        f3(arr, scalar) = f2(arr, scalar)[2]

        primals1 = ([1, 1], 2)
        Δs1 = ([2, 3], 5)
        primals2 = ([1, 2], 1)
        Δs2 = ([1, -2], 1)
        primals3 = ([5, 2], -1)
        Δs3 = ([-3, 1], 0)

        for (primals, Δs) in [(primals1, Δs1), (primals2, Δs2), (primals3, Δs3)]
            for test_deltas in (false, true)
                if test_deltas
                    primals = StochasticAD.structural_map(float, primals)
                    Δs = StochasticAD.structural_map(float, Δs)
                end
                test_propagate(f2, primals, Δs; test_deltas)
                test_propagate(f3, primals, Δs; test_deltas)
            end
        end

        f4(x) = Base.length(repr(x))

        for (primals, Δs) in [(2, 11), (([3, 14],), ([14, -152],))]
            test_propagate(f4, primals, Δs)
        end

        function f5(arr)
            if arr == [1, 2]
                arr .+= 1
            else
                arr .-= 1
            end
        end

        # Tests for f6 skipped (would break)
        for (primals, Δs) in [([1, 2], [1, -1]), ([2, 4], [-1, -2]), ([2, 4], [-1, -1])]
            @test_skip "propagate f5"
            # test_propagate(f5, primals, Δs)
        end

        f6(blob, arr) = blob, f5(arr)

        # Tests for f6 missing (would break)
        @test_skip "propagate f6"

        function f7(mat, scalar)
            return mat * scalar, scalar + sum(mat)
        end

        test_propagate(f7, (rand(2, 2), 4.0), (rand(2, 2), 1.0); test_deltas = true)

        function stoch_trip(val::Real, inf_pert::Real, fin_pert::Real, prob::Real)
            Δs = StochasticAD.similar_new(StochasticAD.create_Δs(PrunedFIsBackend(), Int),
                fin_pert, prob)
            StochasticAD.StochasticTriple{0}(val, inf_pert, Δs)
        end

        f8(x) = x + stoch_trip(1., 0.1, 10., 100.)
        f8(x::StochasticAD.StochasticTriple) = StochasticAD.propagate(f8, x)
        f8() = (x = stoch_trip(2., 0., 20., 100.); f8(x))
        samples = [f8() for _ in 1:10]
        for s in samples
            @test StochasticAD.value(s) == 3.
            @test length(perturbations(s)) == 1
            @test perturbations(s)[1].weight == 200.
        end
        # check that the Δ is sometimes 10 and sometimes 20,
        # which requires the Δs of both the added triples to be taken into account
        Δs = [perturbations(s)[1].Δ for s in samples]
        @test 10. in Δs
        @test 20. in Δs

        function f9(value_1, value_2, rand_var)
            if value_1 < value_2
                return (value_1 + rand(rand_var), value_2)
            else
                return (value_1, value_2 + rand(rand_var))
            end
        end

        propagate_f9(value_1, value_2, rand_var) = StochasticAD.propagate((v1, v2) -> f(v1, v2, rand_var), value_1, value_2)

        f9(value_1::StochasticTriple, value_2, rand_var) = propagate_f9(value_1, value_2, rand_var)
        f9(value_1, value_2::StochasticTriple, rand_var) = propagate_f9(value_1, value_2, rand_var)
        f9(value_1::StochasticTriple, value_2::StochasticTriple, rand_var) = propagate_f9(value_1, value_2, rand_var)

        function g(p)
            rand_var = Bernoulli(p)
            value_1 = 0
            value_2 = 2
            for _ in 1:10
                value_1, value_2 = f9(value_1, value_2, rand_var)
            end
            return value_1, value_2
        end

        N = 100
        derivs = [derivative_estimate(p -> sum(g(p)), 0.5)]
        standard_error = std(derivs) / sqrt(N)
        estimate = mean(derivs)
        expected_value = 10.01  # obtained by running for N = 1E6
        @test estimate - 5standard_error ≤ expected_value ≤ estimate + 5standard_error

    end
end

@testset "zero'ing of Inf/NaN (#79)" begin
    st = stochastic_triple(0.5)
    st_zero = zero(1 / zero(st))
    @test iszero(StochasticAD.value(st_zero))
    @test iszero(StochasticAD.delta(st_zero))
end

@testset "smooth_triple" begin
    f(p) = sum(rand(Bernoulli(p)) * i for i in 1:100)
    f2(p) = sum(smooth_triple(rand(Bernoulli(p))) * i for i in 1:100)
    p = 0.6
    f_est = mean(derivative_estimate(f, p) for i in 1:10000)
    f2_est = mean(derivative_estimate(f2, p) for i in 1:10000)
    @test f_est≈f2_est rtol=5e-2
end

@testset "No unnecessary float promotion" begin
    f(p) = rand(Bernoulli(p))^2
    st = stochastic_triple(f, 0.5)
    @test StochasticAD.valtype(st) == typeof(convert(Signed, f(0.5)))
end
