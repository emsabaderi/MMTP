# Setup file for Parallel Tempering with ChildProcesses in Pigeons

#===================SETUP ENVIRONMENT=====================#
# %% Version info
cd(expanduser("~/.julia/dev/MMTP"))
# run(`julia --version`)

using Pkg
Pkg.activate(".")
# Pkg.instantiate()
# Pkg.status()

# %% General Imports
using Serialization, Revise

import PrettyTables: pretty_table

include("utils/utils.jl");

# %%
#========================DATA TRANSFORMATIONS=============================#
# %% packages
using AxisKeys, DataFrames

import CSV: read

# %% load data
dat = read("assets/data/FS2000_data.csv", DataFrame);
# dat |> describe |> pretty_table

# %% Transform to KeyedArray
data = KeyedArray(Array(dat)',
    Variable=Symbol.("log_" .* names(dat)),
    Time=1:size(dat, 1)
);

# %% logs
data = log.(data);

# %% declare observables
observables = sort(Symbol.("log_" .* names(dat)));
data = data(observables, :);

# %%
#========================INITIALIZE MACROMODEL=============================#
import MacroModelling

# %% define model
MacroModelling.@model FS2000 begin
    dA[0] = exp(γ + z_e_a * e_a[x])

    log(m[0]) = (1 - ρ) * log(mst) + ρ * log(m[-1]) + z_e_m * e_m[x]

    -P[0] / (c[1] * P[1] * m[0]) + β * P[1] * (α * exp(-α * (γ + log(e[1]))) * k[0]^(α - 1) * n[1]^(1 - α) + (1 - δ) * exp(-(γ + log(e[1])))) / (c[2] * P[2] * m[1]) = 0

    W[0] = l[0] / n[0]

    -(ψ / (1 - ψ)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0

    R[0] = P[0] * (1 - α) * exp(-α * (γ + z_e_a * e_a[x])) * k[-1]^α * n[0]^(-α) / W[0]

    1 / (c[0] * P[0]) - β * P[0] * (1 - α) * exp(-α * (γ + z_e_a * e_a[x])) * k[-1]^α * n[0]^(1 - α) / (m[0] * l[0] * c[1] * P[1]) = 0

    c[0] + k[0] = exp(-α * (γ + z_e_a * e_a[x])) * k[-1]^α * n[0]^(1 - α) + (1 - δ) * exp(-(γ + z_e_a * e_a[x])) * k[-1]

    P[0] * c[0] = m[0]

    m[0] - 1 + d[0] = l[0]

    e[0] = exp(z_e_a * e_a[x])

    y[0] = k[-1]^α * n[0]^(1 - α) * exp(-α * (γ + z_e_a * e_a[x]))

    gy_obs[0] = dA[0] * y[0] / y[-1]

    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]

    log_gy_obs[0] = log(gy_obs[0])

    log_gp_obs[0] = log(gp_obs[0])

end

# %% define parameters
MacroModelling.@parameters FS2000 begin
    α = 0.356
    β = 0.993
    γ = 0.0085
    mst = 1.0002
    ρ = 0.129
    ψ = 0.65
    δ = 0.01
    z_e_a = 0.035449
    z_e_m = 0.008862
end

# %%
#============================SET UP TURING (NUTS) SAMPLER=============================#
# %% Sampling imports
import Turing
import Turing: NUTS

import MacroModelling: Beta, Normal, InverseGamma

# %% define prior dists
# MacroModelling instead of Distributions because parameter distribution moments we have need transforming.
# Distributions expects α and β for Beta dists. We have μ and σ. MacroModelling.Beta transforms them, etc...

# %% Specify prior distributions
prior_dists = [
    Beta(0.356, 0.02, μσ=true),                         # α
    Beta(0.993, 0.002, μσ=true),                        # β
    Normal(0.0085, 0.003),                              # γ
    Normal(1.0002, 0.007),                              # mst
    Beta(0.129, 0.223, 1e-4, 1.0 - 1e-4, μσ=true),    # ρ
    Beta(0.65, 0.05, μσ=true),                          # ψ
    Beta(0.01, 0.005, μσ=true),                         # δ
    InverseGamma(0.035449, Inf, μσ=true),               # z_e_a
    InverseGamma(0.008862, Inf, μσ=true)                # z_e_m
];
