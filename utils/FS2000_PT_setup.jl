# Setup file for Parallel Tempering with ChildProcesses in Pigeons

#===================SETUP ENVIRONMENT=====================#
# %% Version info
# cd(expanduser("~/.julia/dev/MMTP"))
# run(`julia --version`)

# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# Pkg.resolve()
# Pkg.status()

# %% General Imports
# include("utils.jl");

# %%
#========================DATA TRANSFORMATIONS=============================#
# %% imports
import CSV: read
using AxisKeys, DataFrames

# %% load data
dat = read("assets/data/FS2000_data.csv", DataFrame)

# %% Transform to KeyedArray
data = KeyedArray(Array(dat)',
    Variable=Symbol.(names(dat)),
    Time=1:size(dat, 1)
);

# %%
#========================INITIALIZE MACROMODEL=============================#
# %% imports
import MacroModelling as MM

# %% define model
MM.@model FS2000 begin
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
MM.@parameters FS2000 begin
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
# %% Setup prior distributions array
# import MacroModelling: Beta, Normal, InverseGamma

# %% define prior dists
# MacroModelling instead of Distributions because parameter distribution moments we have need transforming.
# Distributions expects α and β for Beta dists. We have μ and σ. MacroModelling.Beta transforms them, etc...

# %% Specify prior distributions
import Turing
import MacroModelling: Beta, Normal, InverseGamma

prior_dists = [
    Beta(0.356, 0.02, μσ=true),                         # α
    Beta(0.993, 0.002, μσ=true),                        # β
    Normal(0.0085, 0.003),                              # γ
    Normal(1.0002, 0.007),                              # mst
    Beta(0.129, 0.223, 1e-4, 1.0 - 1e-4, μσ=true),      # ρ
    Beta(0.65, 0.05, μσ=true),                          # ψ
    Beta(0.01, 0.005, μσ=true),                         # δ
    InverseGamma(0.035449, Inf, μσ=true),               # z_e_a
    InverseGamma(0.008862, Inf, μσ=true)                # z_e_m
];

# %%
#============================SET UP TURING AND PIGEONS SAMPLER=============================#
# %% Sampler imports
import DynamicPPL as DPPL

# %% Define PT-compatible Sampler function
Turing.@model function FS2000_loglik_func(prior_dists, data, m, on_loglik_failure; verbose=false)
    parameters ~ Turing.arraydist(prior_dists)

    if DPPL.leafcontext(__context__) !== DPPL.PriorContext()
        llh = MM.get_loglikelihood(m, data, parameters, on_failure_loglikelihood=on_loglik_failure)

        if verbose
            @info "Loglikelihood: $llh and prior llh: $(Turing.logpdf(Turing.arraydist(prior_dists), parameters)) with params $parameters"
        end

        Turing.@addlogprob! llh
    end
end

failure_pen = -floatmax(Float64) + 1e10
FS2000_loglik_pop = FS2000_loglik_func(prior_dists, data, FS2000, failure_pen)

# %%
#============================PIGEONS INITIALIZATION=============================#
# %% imports
using Random
import Pigeons

# %% Pigeons logpotential
init_params = FS2000.parameter_values
FS2000_PT_lp = Pigeons.TuringLogPotential(FS2000_loglik_pop)
typeof_FS2000_PT_LP = typeof(FS2000_PT_lp)

# %% Pigeons initialization
function Pigeons.initialization(target::typeof_FS2000_PT_LP, rng::AbstractRNG, _::Int64)
    result = DPPL.VarInfo(rng, target.model, DPPL.SampleFromPrior(), DPPL.PriorContext())

    result = DPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end
