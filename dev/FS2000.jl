#===================SETUP ENVIRONMENT=====================#
# %% Version info
cd(expanduser("~/.julia/dev/MMTP"))
run(`julia --version`)

using Pkg
Pkg.activate(".")
Pkg.instantiate()
# Pkg.resolve()
# Pkg.status()

# %% Imports
using Serialization, Revise
using AxisKeys, DataFrames
using Random

import MacroModelling as MM
import Turing
import DynamicPPL as DPPL
import Pigeons
include("utils/FS2000_PT_setup.jl")
include("utils/utils.jl")

# %%
#============================BENCHMARK: NUTS=============================#
# %% NUTS Sampler (no parallel tempering)
NUTS_n_samples = 1000
sample_method = NUTS()
error_rng = -floatmax(Float64) + 1e10
FS2000_loglik_pop = FS2000_loglik_func(prior_dists, data, FS2000, error_rng)

FS2000_NUTS_chain = cache_or_compute("assets/cache/FS2000_NUTS_chain.jls") do
    Turing.sample(FS2000_loglik_pop,
        sample_method,
        NUTS_n_samples,
        initial_params=FS2000.parameter_values)

end

# %% NUTS plot
paramlist = MM.get_parameters(FS2000)
FS2000_NUTS_chain_renamed = Turing.replacenames(
    FS2000_NUTS_chain,
    Dict(["parameters[$i]" for i in 1:length(paramlist)] .=> MM.get_parameters(FS2000))
)

FS2000_NUTS_chain_plot = plot(FS2000_NUTS_chain_renamed)
savefig(FS2000_NUTS_chain_plot, "assets/plots/FS2000_NUTS_chain_plot.png")

# %%
#============================PARALLEL TEMPERING STANDALONE (SINGLE PROCESS)=============================#
