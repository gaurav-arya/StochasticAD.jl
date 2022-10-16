include("core.jl")
include("model.jl")
using Plots, LaTeXStrings
using Random

Random.seed!(seed)
# Comparison of the variance of the particle filter with and without differentiating the resampling step *as a function of the time steps*.
vars_pf = []
vars_pf_biased = []
Ts = 5:5:30
for T in Ts
    # Random.seed!(seed) is fixed in model!
    θtrue, xs, ys, stochastic_model, kalman_filter, particle_filter = generate_system(d, T)
    xs, ys = ParticleFilterCore.simulate_single(stochastic_model, θtrue)
    particle_filter = ParticleFilterCore.ParticleFilter(m, stochastic_model, ys,
                                                        ParticleFilterCore.sample_stratified)
    ### compute var of gradients
    # Gradient of the particle filter *with* differentiation of the resampling step
    var_pf = @time var([ParticleFilterCore.forw_grad(θtrue, particle_filter) for i in 1:100])
    # Gradient of the particle filter *without* differentiation of the resampling step
    var_pf_biased = @time var([ParticleFilterCore.forw_grad_biased(θtrue, particle_filter)
                               for i in 1:100])

    push!(vars_pf, var_pf)
    push!(vars_pf_biased, var_pf_biased)
end

@show vars_pf
@show vars_pf_biased

# pick an arbitrary coordinate
index = 1 # take derivative with respect to first parameter
fig = plot(Ts, getindex.(vars_pf, index), color = 1, label = "unbiased", size = (300, 250),
           xlabel = L"n", ylabel = "variance", legend = :topleft, y_scale = :log)
scatter!(Ts, getindex.(vars_pf, index), color = 1, label = false)
plot!(Ts, getindex.(vars_pf_biased, index), color = 2, label = "biased")
scatter!(Ts, getindex.(vars_pf_biased, index), color = 2, label = false)
display(fig)
savefig(fig, "particle_filter_variance_steps.pdf")

# Comparison of the variance of the particle filter with and without differentiating the resampling step *as a function of the system size*.
vars_pf = []
vars_pf_biased = []
ds = 2:1:6
for d in ds
    # Random.seed!(seed) is fixed in model!
    θtrue, xs, ys, stochastic_model, kalman_filter, particle_filter = generate_system(d, 10)
    xs, ys = ParticleFilterCore.simulate_single(stochastic_model, θtrue)
    particle_filter = ParticleFilterCore.ParticleFilter(m, stochastic_model, ys,
                                                        ParticleFilterCore.sample_stratified)
    ### compute var of gradients
    # Gradient of the particle filter *with* differentiation of the resampling step
    var_pf = @time var([ParticleFilterCore.forw_grad(θtrue, particle_filter) for i in 1:50])
    # Gradient of the particle filter *without* differentiation of the resampling step
    var_pf_biased = @time var([ParticleFilterCore.forw_grad_biased(θtrue, particle_filter)
                               for i in 1:50])

    push!(vars_pf, var_pf)
    push!(vars_pf_biased, var_pf_biased)
end

fig = plot(ds, getindex.(vars_pf, index), color = 1, label = "unbiased", size = (300, 250),
           xlabel = L"d", ylabel = "variance", legend = :topleft, y_scale = :log)
scatter!(ds, getindex.(vars_pf, index), color = 1, label = false)
plot!(ds, getindex.(vars_pf_biased, index), color = 2, label = "biased")
scatter!(ds, getindex.(vars_pf_biased, index), color = 2, label = false)
display(fig)
savefig(fig, "particle_filter_variance_size.pdf")
