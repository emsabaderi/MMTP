#===================SETUP ENVIRONMENT=====================#
# %% Version info
cd(expanduser("~/.julia/dev/MMTP"))
run(`julia --version`)

using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.resolve()
Pkg.status()

# %% Imports
using Revise, BenchmarkTools

import CondaPkg
import Serialization: serialize, deserialize
import Turing
import Turing: NUTS
import MacroModelling as MM
import StatsPlots: plot, savefig
import Pigeons
import MCMCChains as MCMCC

CondaPkg.resolve()
include("utils/utils.jl")
include("utils/FS2000_PT_setup.jl")

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
rename_map = Dict("parameters[$i]" => string(p) for (i, p) in enumerate(paramlist))

FS2000_NUTS_chain_renamed = MCMCC.replacenames(FS2000_NUTS_chain, rename_map)

MCMCC.summarize(FS2000_NUTS_chain_renamed)

# %%
FS2000_NUTS_chain_plot = plot(FS2000_NUTS_chain_renamed)
savefig(FS2000_NUTS_chain_plot, "assets/plots/FS2000_NUTS_chain_plot.png")

# %%
#============================PARALLEL TEMPERING=============================#
# %% Parallel Tempered sampler (single thread - no MT)
PIGEONS_SEED = 123

# %% Parallel Tempering Sampler
FS2000_PT = pt_cache_or_compute("assets/cache/FS2000_PT_chain") do
    Pigeons.pigeons(
        target=FS2000_PT_lp,
        record=[
            Pigeons.traces;
            Pigeons.round_trip;
            Pigeons.record_default()
        ],
        n_chains=10,
        n_rounds=7,
        seed=PIGEONS_SEED,
        checkpoint=true,
        on=Pigeons.ChildProcess(
            n_local_mpi_processes=4,
            n_threads=1,
            dependencies=[abspath("utils/FS2000_PT_setup.jl")]
        )
    )
end
# %%
FS2000_PT |> typeof

# %% PT chain analysis
# FS2000_PT_load = Pigeons.load(FS2000_PT)
FS2000_PT_chain = MCMCC.Chains(FS2000_PT)
FS2000_PT_chain_renamed = Turing.replacenames(FS2000_PT_chain, Dict(["parameters[$i]" for i in 1:length(paramlist)] .=> MM.get_parameters(FS2000)))

# %% PT sampler plot
FS2000_PT_chain_plot = plot(FS2000_PT_chain_renamed)
savefig(FS2000_PT_chain_plot, "assets/plots/FS2000_PT_chain_plot.png")

# %% PT restarts, barrier plot
using Interpolations
Λ = FS2000_PT_load.shared.tempering.communication_barriers.globalbarrier
restarts = Pigeons.n_round_trips(FS2000_PT_load)
localbar = FS2000_PT_load.shared.tempering.communication_barriers.localbarrier

# Extract the interpolation (single-field struct)
itp = getfield(localbar, first(fieldnames(typeof(localbar))))

# Sample it on a dense β grid for plotting
βs = range(0, 1, length=200)
cum_barrier = itp.(βs)                                            # cumulative
local_barrier_density = [Interpolations.gradient(itp, β)[1] for β in βs]  # derivative

# %% plot the local barrier — your "where does the difficulty live" curve
p1 = plot(βs, cum_barrier;
    xlabel="β  (tempering parameter)",
    ylabel="cumulative barrier  Λ(β)",
    title="FS2000 — cumulative communication barrier",
    legend=false, lw=2)

p2 = plot(βs, local_barrier_density;
    xlabel="β  (tempering parameter)",
    ylabel="local barrier  λ(β)",
    title="FS2000 — local communication barrier",
    legend=false, lw=2)

plt = plot(p1, p2, layout=(1, 2), size=(1200, 400))
savefig(plt, "assets/plots/FS2000_PT_barriers.png")
# %%
