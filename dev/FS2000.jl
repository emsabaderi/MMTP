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
using Revise

import CondaPkg
import Serialization: serialize, deserialize
import Turing
import Turing: NUTS
import MacroModelling as MM
import StatsPlots: plot, savefig
import Pigeons
import MCMCChains as MCMCC

CondaPkg.resolve()
include("utils.jl")
include("utils/FS2000_PT_setup.jl")
# include("utils/utils.jl")

# %%
#============================BENCHMARK: NUTS=============================#
# %% imports

# %% NUTS Sampler (no parallel tempering)
sample_method = NUTS()
NUTS_n_samples = 1000

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
#============================PARALLEL TEMPERING=============================#
# %% Parallel Tempered sampler (single thread - no MT)
PIGEONS_SEED = 123

# %% Parallel Tempering Sampler
FS2000_PT = cache_or_compute("assets/cache/FS2000_PT_chain.jls") do
    Pigeons.pigeons(
        target=FS2000_PT_lp,
        record=[
            Pigeons.traces;
            Pigeons.round_trip;
            Pigeons.record_default()
        ],
        n_chains=13,
        n_rounds=12,
        seed=PIGEONS_SEED,
        checkpoint=true,
        on=Pigeons.ChildProcess(
            n_local_mpi_processes=4,
            n_threads=1,
            dependencies=[abspath("utils/FS2000_PT_setup.jl")]
        )
    )
end

# %% PT chain analysis
FS2000_PT_chain = MCMCC.Chains(FS2000_PT)
FS2000_PT_chain_renamed = Turing.replacenames(FS2000_PT_chain, Dict(["parameters[$i]" for i in 1:length(paramlist)] .=> get_parameters(FS2000_PT)))

# %% PT plot
FS2000_PT_chain_plot = plot(FS2000_PT_chain_renamed)
savefig(FS2000_PT_chain_plot, "assets/plots/FS2000_PT_chain_plot.png")
