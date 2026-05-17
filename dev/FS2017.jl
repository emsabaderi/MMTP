#============================SETUP=============================#
# %% version info
cd("..")
pwd()
run(`julia --version`)
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.status()

# %% General Imports
using Serialization, Revise

import PrettyTables: pretty_table

include("utils/MMTutils.jl");

# %%
#============================DATA TRANSFORMATIONS=============================#
# %% packages
using AxisKeys, DataFrames

import CSV: read

# %% load DataFrame
dat = read("assets/data/FS2017_data.csv", DataFrame);
dat |> describe |> pretty_table

# %% Transform to KeyedArray
data = KeyedArray(Array(dat)', Variable=Symbol.(names(dat)), Time=1:size(dat, 1))

# %% declare observables
observables = sort(Symbol.(names(dat)));
data = data(observables, :)

# %%
#============================MACROMODEL=============================#
# %%
import MacroModelling

# %% define model
MacroModelling.@model FS2017 begin
    yhat[0] = yhat[1] - 1 / τ * (rhat[0] - πhat[1] - zhat[1]) + ghat[0] - ghat[1]
    πhat[0] = (1 / (1 + rA / 400)) * πhat[1] + κ * (yhat[0] - ghat[0])

    rhat[0] = ρr * rhat[-1] + (1 - ρr) * ψ1 * πhat[0] + σr * ϵr[x]
    zhat[0] = ρz * zhat[-1] + σz * ϵz[x]
    ghat[0] = ρg * ghat[-1] + σg * ϵg[x]

    ygr[0] = γQ + (yhat[0] - yhat[-1] + zhat[0])
    infl[0] = πA + 4 * πhat[0]
    int[0] = rA + πA + 4γQ + 4 * rhat[0]

end

# %% define parameters
MacroModelling.@parameters FS2017 begin
    rA = 0.25
    πA = 5.0
    γQ = 0.2

    τ = 1.50
    κ = 0.25
    ψ1 = 1.25
    # ψ2 = 0.5
    ρr = 0.25

    ρg = 0.25
    ρz = 0.25
    σr = 0.251326
    σg = 1.003314
    σz = 0.506657
end

# %%
#============================SET UP SAMPLER=============================#
# %% Sampling imports
import Turing
import DynamicPPL
import Pigeons
import MacroModelling: Gamma, Normal, InverseGamma
import Distributions: Uniform

using Random

# %% Specify prior distributions
prior_dists = [
    Gamma(0.50, 0.50, μσ=true),                  # rA
    Gamma(7.00, 2.00, μσ=true),                  # πA
    Normal(0.40, 0.20),                          # γQ
    Gamma(2.00, 0.50, μσ=true),                  # τ
    Uniform(0.00, 1.00),                         # κ
    Gamma(1.50, 0.25, μσ=true),                  # ψ1
    # Gamma(0.50, 0.25, μσ=true),                  # ψ2
    Uniform(0.00, 1.00),                         # ρr
    Uniform(0.00, 1.00),                         # ρg
    Uniform(0.00, 1.00),                         # ρz
    InverseGamma(0.501326, 0.262055, μσ=true),   # σr
    InverseGamma(1.253314, 0.655136, μσ=true),   # σg
    InverseGamma(0.626657, 0.327568, μσ=true)   # σz
];

# %% define sampling model
Turing.@model function FS2017_loglik_func(data, m, on_loglik_fail)
    parameters ~ Turing.arraydist(prior_dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        Turing.@addlogprob! MacroModelling.get_loglikelihood(m, data, parameters, on_failure_loglikelihood=on_loglik_fail)
    end
end

# %% Pigeons initialization

FS2017_lp = Pigeons.TuringLogPotential(FS2017_loglik_func(data, FS2017, -floatmax(Float64) + 1e10))

init_params = FS2017.parameter_values
PIGEONS_SEED = 30

FS2017_LP = typeof(FS2017_lp)

function Pigeons.initialization(target::FS2017_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())

    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

# %% Pigeons sample

pt = Pigeons.pigeons(target=FS2017_lp, n_rounds=0, n_chains=1, seed=PIGEONS_SEED)

# %%

pt2 = Pigeons.pigeons(
    target=FS2017_lp,
    record=[Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
    n_chains=10,
    n_rounds=10,
    seed=PIGEONS_SEED,
    multithreaded=false
)

# %%

import MCMCChains
using StatsPlots
# %%

samps = MCMCChains.Chains(pt2);
sampsplot = plot(samps)
savefig(sampsplot, "assets/plots/samps.png")
