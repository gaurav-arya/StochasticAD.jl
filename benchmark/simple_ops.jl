module SimpleOpsBenchmark

using BenchmarkTools

using StochasticAD

const suite = BenchmarkGroup()

suite["add"] = BenchmarkGroup()
suite["add_via_propagate_nodeltas"] = BenchmarkGroup()
suite["add_via_propagate"] = BenchmarkGroup()

suite["add"]["original"] = @benchmarkable +(0.5, 0.5)
suite["add_via_propagate_nodeltas"]["original"] = @benchmarkable StochasticAD.propagate(+,
                                                                                        0.5,
                                                                                        0.5)
suite["add_via_propagate"]["original"] = @benchmarkable StochasticAD.propagate(+, 0.5, 0.5;
                                                                               keep_deltas = Val{
                                                                                                 true
                                                                                                 })
for backend in [StochasticAD.PrunedFIs, StochasticAD.PrunedFIsAggressive]
    suite["add"][backend] = @benchmarkable +(st, st) setup=(st = stochastic_triple(0.5;
                                                                                   backend = $backend))
    suite["add_via_propagate_nodeltas"][backend] = @benchmarkable StochasticAD.propagate(+,
                                                                                         st,
                                                                                         st) setup=(st = stochastic_triple(0.5;
                                                                                                                           backend = $backend))
    suite["add_via_propagate"][backend] = @benchmarkable StochasticAD.propagate(+, st, st;
                                                                                keep_deltas = Val{
                                                                                                  true
                                                                                                  }) setup=(st = stochastic_triple(0.5;
                                                                                                                                   backend = $backend))
end

end
