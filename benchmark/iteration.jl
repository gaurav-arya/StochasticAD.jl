"""
In the library we have tried to avoid generated functions, instead reductions from base with the
hope that the iteration will be optimized to be zero-cost.
This suite tests the performance of iteration on small nested structures, which crop up when `propagate` is called
on small structures of scalars.
The `couple` and `combine` operations of FIss, which use iteration, are benchmarked.
"""
module IterationBenchmark

using BenchmarkTools
using StochasticAD
using StaticArrays

const suite = BenchmarkGroup()

# Examples consist of flat and non-flat versions of structures, to test zero-cost iteration.
tups = Dict("easy" => (ntuple(identity, 3), (1, (2, 3))),
    "hard" => (ntuple(identity, 9), (1, (2, 3), (4, (5, (6, 7, 8), 9)))))
SAs = Dict("easy" => (SA[1, 2, 3], (1, SA[2, 3])),
    "hard" => (SA[1, 2, 3, 4, 5, 6, 7, 8, 9],
        (1, SA[2, 3], (4, (5, SA[6, 7, 8], 9)))))

for (setname, set) in (("tups", tups), ("SAs", SAs))
    suite[setname] = BenchmarkGroup()
    setsuite = suite[setname]
    for case in ["easy", "hard"]
        casesuite = setsuite[case] = BenchmarkGroup()
        for isflat in [false, true]
            flatsuite = casesuite[isflat ? "flat" : "not flat"] = BenchmarkGroup()
            values = set[case][isflat ? 1 : 2]
            flatsuite["make_iterate_values"] = @benchmarkable StochasticAD.structural_iterate($values)
            iter_values = StochasticAD.structural_iterate(values)
            flatsuite["foldl_values"] = @benchmarkable foldl(+, $(iter_values))
            flatsuite["iterate_values"] = @benchmarkable for i in $(iter_values)
            end
            for backend in [PrunedFIsBackend(), PrunedFIsAggressiveBackend()]
                FIs_suite = flatsuite[backend] = BenchmarkGroup()
                Δs = StochasticAD.create_Δs(backend, Int)
                Δs1 = StochasticAD.similar_new(Δs, 1, 1)
                Δs_all = StochasticAD.structural_map(x -> map(Δ -> x, Δs1), values)
                FIs_suite["make_iterate_Δs"] = @benchmarkable StochasticAD.structural_iterate($Δs_all)
                # We don't interpolate backend directly in below (i.e. do $FIs) because string interpolating a type
                # seems to lead to slow benchmarks.
                FIs_suite["couple_same"] = @benchmarkable StochasticAD.couple(typeof($Δs),
                    $Δs_all)
                FIs_suite["combine_same"] = @benchmarkable StochasticAD.combine(typeof($Δs),
                    $Δs_all)
            end
        end
    end
end

end
