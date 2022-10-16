include("core.jl")
using Plots, LaTeXStrings
using Statistics
using StochasticAD

stds_triple = Float64[]
stds_score = Float64[]
stds_score_baseline = Float64[]
for (n, p) in zip(RandomWalkCore.n_range, RandomWalkCore.p_range)
    std_triple = std(derivative_estimate(p -> RandomWalkCore.fX(p, n), p)
                     for i in 1:(RandomWalkCore.nsamples))
    std_score = std(RandomWalkCore.score_fX_deriv(p, n, 0.0)
                    for i in 1:(RandomWalkCore.nsamples))
    avg = mean(RandomWalkCore.fX(p, n) for i in 1:10000)
    std_score_baseline = std(RandomWalkCore.score_fX_deriv(p, n, avg)
                             for i in 1:(RandomWalkCore.nsamples))
    push!(stds_triple, std_triple)
    push!(stds_score, std_score)
    push!(stds_score_baseline, std_score_baseline)
end

@show stds_triple
@show stds_score
@show stds_score_baseline

fig = plot(RandomWalkCore.n_range, stds_score, color = 2, label = "score-function",
           size = (300, 250),
           xlabel = L"n", ylabel = "standard deviation", legend = :topleft)
scatter!(RandomWalkCore.n_range, stds_score, color = 2, label = false)
plot!(RandomWalkCore.n_range, stds_score_baseline, color = 3,
      label = "score-function w/ CV")
scatter!(RandomWalkCore.n_range, stds_score_baseline, color = 3, label = false)
plot!(RandomWalkCore.n_range, stds_triple, color = 1, label = "stochastic triples")
scatter!(RandomWalkCore.n_range, stds_triple, color = 1, label = false)
display(fig)
savefig(fig, "random_walk.pdf")

# same for stdiance
stds_triple = []
stds_score = []
stds_score_baseline = []
for (n, p) in zip(RandomWalkCore.n_range, RandomWalkCore.p_range)
    std_triple = std(derivative_estimate(p -> RandomWalkCore.fX(p, n), p)
                     for i in 1:(RandomWalkCore.nsamples))
    std_score = std(RandomWalkCore.score_fX_deriv(p, n, 0.0)
                    for i in 1:(RandomWalkCore.nsamples))
    avg = mean(RandomWalkCore.fX(p, n) for i in 1:10000)
    std_score_baseline = std(RandomWalkCore.score_fX_deriv(p, n, avg)
                             for i in 1:(RandomWalkCore.nsamples))
    push!(stds_triple, std_triple)
    push!(stds_score, std_score)
    push!(stds_score_baseline, std_score_baseline)
end

@show stds_triple
@show stds_score
@show stds_score_baseline

fig = plot(RandomWalkCore.n_range, stds_score, color = 2, label = "score-function",
           size = (300, 250),
           xlabel = L"n", ylabel = "standard deviation", legend = :topleft)
scatter!(RandomWalkCore.n_range, stds_score, color = 2, label = false)
plot!(RandomWalkCore.n_range, stds_score_baseline, color = 3,
      label = "score-function w/ CV")
scatter!(RandomWalkCore.n_range, stds_score_baseline, color = 3, label = false)
plot!(RandomWalkCore.n_range, stds_triple, color = 1, label = "stochastic triples")
scatter!(RandomWalkCore.n_range, stds_triple, color = 1, label = false)
display(fig)
savefig(fig, "compare_score.png")
