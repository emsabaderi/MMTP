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
FS2000_loglik = FS2000_loglik_func(prior_dists, data, FS2000)
my_turing_target = TuringLogPotential(FS2000_loglik)

# %% (your custom initialization stays as-is, no changes)

# Sample
rm("assets/cache/pigeons_chain.jls", force=true)
pigeons_chain = cache_or_compute("assets/cache/pigeons_chain.jls") do
    pt = pigeons(
        target=my_turing_target,
        explorer=AutoMALA(),       # gradient-based, avoids prior singularity
        multithreaded=false,            # MacroModelling isn't re-entrant
        record=[traces; record_default()]
    )
    Chains(pt)
end

# %%
