include("core.jl")
using Plots
using Statistics
using BenchmarkTools

p = 0.5
_, board, history = stochastic_triple(p -> GoLCore.play(p; log = true), p)

anim1 = @animate for (i, board) in enumerate(history)
    heatmap(collect(StochasticAD.value.(board)), title = "time $i", clim = (-1, 1),
            c = :grays)
end
anim2 = @animate for (i, board) in enumerate(history)
    heatmap(collect(StochasticAD.derivative_contribution.(board)), title = "time $i", clim = (-1, 1), c = :grays)
end

gif(anim1, "game.gif", fps = 15)
gif(anim2, "perturbation.gif", fps = 15)
fig1 = heatmap(collect(StochasticAD.value.(board)), clim = (-1, 1), c = :grays)
fig2 = heatmap(collect(derivative_contribution.(board)), clim = (-1, 1), c = :grays) # TODO: graph perturbed values instead of derivative contribution
savefig(fig1, "board.png")
savefig(fig2, "pertubation.png")
