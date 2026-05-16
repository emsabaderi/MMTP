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
    rA = 025
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
import Turing: NUTS

import MacroModelling: Gamma, Normal, InverseGamma
import Distributions: Uniform

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
Turing.@model function FS2017_loglik_func(prior_dists, data, m; verbose=false)
    parameters ~ Turing.arraydist(prior_dists)

    Turing.@addlogprob! MacroModelling.get_loglikelihood(m, data, parameters)
end

# %% Specify sampler parameters
FS2017_loglik = FS2017_loglik_func(prior_dists, data, FS2017);
n_samples = 1000
sample_method = NUTS()

# %% sample
FS2017_CHAIN = cache_or_compute("assets/cache/FS2017_CHAIN.jls") do
    Turing.sample(FS2017_loglik,
        sample_method,
        n_samples,
        initial_params=FS2017.parameter_values)
end

# %%
#============================MAKING SENSE OF POSTERIOR=============================#
# %% Inspection Imports
import StatsPlots: plot, savefig
import MacroModelling: get_parameters
import Turing: replacenames

# %% replacing posterior names
paramlist = get_parameters(FS2017)
FS2017_CHAIN_rn = replacenames(FS2017_CHAIN, Dict(["parameters[$i]" for i in 1:length(paramlist)] .=> get_parameters(FS2017)))

# %% plot
FS2017_CHAIN_plot = plot(FS2017_CHAIN_rn)

# %%
savefig(FS2017_CHAIN_plot, "assets/plots/FS2017_CHAIN_plot.png")

# %%
#============================PARALLEL TEMPERING=============================#
# %% Import Pigeons stuff

import DynamicPPL, Pigeons

# %% specify Pigeons-compatible likelihood function
DynamicPPL.@model function FS2017_pigeons_loglik_func(prior_dists, data, m; verbose=false)
    parameters ~ Turing.arraydist(prior_dists)
    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext()
        DynamicPPL.@addlogprob! MacroModelling.get_loglikelihood(m, data, parameters)
    end
end
# %%

FS2017_pigeons_loglik = FS2017_pigeons_loglik_func(prior_dists, data, FS2017);
FS2017_pigeons_target = Pigeons.TuringLogPotential(FS2017_pigeons_loglik)
FS2017_pigeons = Pigeons.pigeons(target=FS2017_pigeons_target)

# %%
