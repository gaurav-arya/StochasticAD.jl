include("core.jl")
include("model.jl")
using Plots, LaTeXStrings
using Random

# Comparison of the derivative of the particle filter with and without differentiating the resampling step.

### compute gradients
Random.seed!(seed)
X = [ParticleFilterCore.forw_grad(θtrue, particle_filter) for i in 1:1000] # gradient of the particle filter *with* differentiation of the resampling step
Random.seed!(seed)
Xbiased = [ParticleFilterCore.forw_grad_biased(θtrue, particle_filter) for i in 1:1000] # Gradient of the particle filter *without* differentiation of the resampling step
# pick an arbitrary coordinate
index = 1 # take derivative with respect to first parameter (2-dimensional example has a rotation matrix with four parameters in total)
# plot histograms for the sampled derivative values
fig = plot(normalize(fit(Histogram, getindex.(X, index), nbins = 50), mode = :pdf),
           legend = false) # ours
plot!(normalize(fit(Histogram, getindex.(Xbiased, index), nbins = 50), mode = :pdf)) # biased
vline!([mean(X)[index]], color = 1)
vline!([mean(Xbiased)[index]], color = 2)
# add derivative of differentiable Kalman filter as a comparison
XK = ParticleFilterCore.forw_grad_Kalman(θtrue, kalman_filter)
vline!([XK[index]], color = "black")

display(fig)
savefig(fig, "tails.pdf")
