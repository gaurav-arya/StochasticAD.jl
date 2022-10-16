using StochasticAD
using Test
using Distributions
using ForwardDiff

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
        (p -> Categorical([0, p^2, 0, 0, 1 - p^2])), # To check that 0's are skipped over
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
