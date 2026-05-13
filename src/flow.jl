# %% Setup environment
using Pkg
cd(joinpath(@__DIR__,".."))
Pkg.activate(".")

# %% prep data
# include("data/dataprep.jl")

# %%
using CSV, DataFrames, AxisKeys, MacroModelling, PrettyTables, Serialization


# %% helper for serialized samples
function cache_or_compute(compute_fn,path::AbstractString; force::Bool=false)
    if !force && isfile(path)
        @info "Loading cached result from $path"
        return deserialize(path)
    end
    @info "Computing fresh result and caching to $path"
    mkpath(dirname(path))
    result = compute_fn()
    serialize(path, result)
    return result
end

# %% load data
dat = CSV.read("assets/data/FS2000_data.csv", DataFrame);
pretty_table(describe(dat))

# %% Transform to KeyedArray
data = KeyedArray(Array(dat)',
    Variable = Symbol.("log_".*names(dat)),
    Time= 1:size(dat,1)
);

# %% logs
data = log.(data);

# %% declare observables
observables = sort(Symbol.("log_".*names(dat)));
data = data(observables,:)
describe(data)

# %% define model
@model FS2000 begin
    dA[0] = exp(γ + z_e_a*e_a[x])

    log(m[0]) = (1 - ρ)*log(mst)  +  ρ*log(m[-1]) + z_e_m*e_m[x]

    - P[0] / (c[1]*P[1]*m[0]) + β*P[1]*(α*exp(-α*(γ + log(e[1])))*k[0] ^ (α - 1)*n[1] ^ (1 - α) + (1 - δ)*exp( - (γ + log(e[1])))) / (c[2]*P[2]*m[1]) = 0

    W[0] = l[0] / n[0]

    - (ψ / (1 - ψ))*(c[0]*P[0] / (1 - n[0])) + l[0] / n[0] = 0

    R[0] = P[0]*(1 - α)*exp( - α*(γ + z_e_a*e_a[x]))*k[-1] ^ α*n[0] ^ ( - α) / W[0]

    1 / (c[0]*P[0]) - β*P[0]*(1 - α)*exp( - α*(γ + z_e_a*e_a[x]))*k[-1] ^ α*n[0] ^ (1 - α) / (m[0]*l[0]*c[1]*P[1]) = 0

    c[0] + k[0] = exp( - α*(γ + z_e_a*e_a[x]))*k[-1] ^ α*n[0] ^ (1 - α) + (1 - δ)*exp( - (γ + z_e_a*e_a[x]))*k[-1]

    P[0]*c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    e[0] = exp(z_e_a*e_a[x])

    y[0] = k[-1] ^ α*n[0] ^ (1 - α)*exp( - α*(γ + z_e_a*e_a[x]))

    gy_obs[0] = dA[0]*y[0] / y[-1]

    gp_obs[0] = (P[0] / P[-1])*m[-1] / dA[0]

    log_gy_obs[0] = log(gy_obs[0])

    log_gp_obs[0] = log(gp_obs[0])

end

# %% define parameters
@parameters FS2000 begin
    α     = 0.356
    β     = 0.993
    γ     = 0.0085
    mst   = 1.0002
    ρ     = 0.129
    ψ     = 0.65
    δ     = 0.01
    z_e_a = 0.035449
    z_e_m = 0.008862
end

# %% Sampling imports
import Turing
import Turing: NUTS, sample, logpdf, replacenames
import DynamicPPL

import ADTypes: AutoZygote, AutoReverseDiff, ForwardDiff
import Zygote

using DynamicPPL, Distributions, MCMCChains, StatsPlots

# %% define prior dists
prior_dists = [
    MacroModelling.Beta(0.356, 0.02, μσ = true),           # \alpha
    MacroModelling.Beta(0.993, 0.002, μσ = true),          # \beta
    MacroModelling.Normal(0.0085, 0.003),                  # \gamma
    MacroModelling.Normal(1.0002, 0.007),                  # mst
    MacroModelling.Beta(0.129, 0.223, μσ = true),          # \rho
    MacroModelling.Beta(0.65, 0.05, μσ = true),            # \psi
    MacroModelling.Beta(0.01, 0.005, μσ = true),           # \delta
    MacroModelling.InverseGamma(0.035449, Inf, μσ = true), # z_e_a
    MacroModelling.InverseGamma(0.008862, Inf, μσ = true)  # z_e_m
];

# %% define sampling model
Turing.@model function FS2000_loglik_func(prior_dists, data, m; verbose=false)

    parameters ~ Turing.arraydist(prior_dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        DynamicPPL.@addlogprob! get_loglikelihood(m, data, parameters)
    end
end

# %% Fit priors, data, model
FS2000_loglik = FS2000_loglik_func(prior_dists, data, FS2000) ;

# %% Sampler: NUTS (cached)
n_samples = 1000
chain_NUTS = cache_or_compute("assets/cache/chain_NUTS.jls") do
    sample(FS2000_loglik, NUTS(), n_samples,
           initial_params = FS2000.parameter_values)
end

# %% Inspecting Posterior
paramlist = get_parameters(FS2000)
chain_NUTS_rn = replacenames(chain_NUTS, Dict(["parameters[$i]" for i in 1:length(paramlist)] .=> get_parameters(FS2000)))
# %%
chain_NUTS_plot = plot(chain_NUTS_rn)

# %%
savefig(chain_NUTS_plot, "assets/plots/chain_NUTS_plot.png")

# %% Sampling: Parallel Tempering with Pigeons.jl
using Pigeons, Random

my_turing_target = TuringLogPotential(FS2000_loglik)

# %%
# pigeons_chain = cache_or_compute("assets/cache/pigeons_chain.jls") do
#     pt = pigeons(
#         target = my_turing_target,
#         multithreaded = false,
#         record = [traces; record_default()]   # capture samples for post-processing
#     )
#     Chains(pt)
# end
#= SliceSampler error. Pigeons logs tell me I should set up my own custom initialization =#

# %%
FS2000TargetType = typeof(my_turing_target)

function Pigeons.initialization(target::FS2000TargetType, rng::AbstractRNG, ::Int64)
    result = DynamicPPL.VarInfo(rng, target.model,
                                DynamicPPL.SampleFromPrior(),
                                DynamicPPL.PriorContext())
    # write calibrated values in CONSTRAINED space first
    for i in eachindex(FS2000.parameter_values)
        Pigeons.update_state!(result, :parameters, i, FS2000.parameter_values[i])
    end
    # then link, which applies the bijector — values now correctly in unconstrained space
    result = DynamicPPL.link(result, target.model)
    return result
end
# %%


Turing.@model function FS2000_loglik_func(prior_dists, data, m; verbose=false)
    parameters ~ Turing.arraydist(prior_dists)
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        ll = try
            get_loglikelihood(m, data, parameters)
        catch e
            convert(eltype(parameters), -Inf)
        end
        DynamicPPL.@addlogprob! ll
    end
end

# %% Rebuild after editing the model:
FS2000_loglik    = FS2000_loglik_func(prior_dists, data, FS2000)
my_turing_target = TuringLogPotential(FS2000_loglik)

# %% (your custom initialization stays as-is, no changes)

# Sample
rm("assets/cache/pigeons_chain.jls", force=true)
pigeons_chain = cache_or_compute("assets/cache/pigeons_chain.jls") do
    pt = pigeons(
        target        = my_turing_target,
        explorer      = AutoMALA(),       # gradient-based, avoids prior singularity
        multithreaded = false,            # MacroModelling isn't re-entrant
        record        = [traces; record_default()]
    )
    Chains(pt)
end

# %%
