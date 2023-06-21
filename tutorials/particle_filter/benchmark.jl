include("core.jl")
include("model.jl")
using Plots, LaTeXStrings
using BenchmarkTools
using Measurements

# Benchmark for primal, forward- and reverse-mode AD of particle sampler

### compute gradients
# secs for how long the benchmark should run, see https://juliaci.github.io/BenchmarkTools.jl/stable/
secs = 10

suite = BenchmarkGroup()
suite["scaling"] = BenchmarkGroup(["grads"])

suite["scaling"]["primal"] = @benchmarkable ParticleFilterCore.log_likelihood(particle_filter,
    θtrue)
suite["scaling"]["forward"] = @benchmarkable ParticleFilterCore.forw_grad(θtrue,
    particle_filter)
suite["scaling"]["backward"] = @benchmarkable ParticleFilterCore.back_grad(θtrue,
    particle_filter)

tune!(suite)
results = run(suite, verbose = true, seconds = secs)

t1 = measurement(mean(results["scaling"]["primal"].times),
    std(results["scaling"]["primal"].times) /
    sqrt(length(results["scaling"]["primal"].times)))
t2 = measurement(mean(results["scaling"]["forward"].times),
    std(results["scaling"]["forward"].times) /
    sqrt(length(results["scaling"]["forward"].times)))
t3 = measurement(mean(results["scaling"]["backward"].times),
    std(results["scaling"]["backward"].times) /
    sqrt(length(results["scaling"]["backward"].times)))
@show t1 t2 t3

ts = (t1, t2, t3) ./ 10^6 # ms
@show ts

BenchmarkTools.save("benchmark_data_" * string(d) * ".json", results)
