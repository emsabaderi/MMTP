# Setup file for Parallel Tempering with ChildProcesses in Pigeons

#===================SETUP ENVIRONMENT=====================#
# %% Version info
# cd(expanduser("~/.julia/dev/MMTP"))
# run(`julia --version`)

using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.resolve()
Pkg.status()

# %% General Imports
# include("utils.jl");
using LinearAlgebra
BLAS.set_num_threads(1)

# %%
#========================DATA TRANSFORMATIONS=============================#
# %% imports, setup
import CSV: read
using AxisKeys, DataFrames

# %% load data
dat = read("assets/data/HS2017_data.csv", DataFrame)
data = KeyedArray(Array(dat)',
    Variable=Symbol.(names(dat)),
    Time=1:size(dat, 1)
);

# %%
#========================INITIALIZE MACROMODEL=============================#
# %% imports
import MacroModelling as MM

# %% define model
MM.@model HS2017 begin

    y[0] = y[1] - (1 / τ) * (r[0] - π[1] - z[1]) + g[0] - g[1]
    π[0] = β * π[1] + κ * (y[0] - g[0])
    r[0] = ρR * r[-1] + (1 - ρR) * ψ * π[0] + σR * ϵR[x]
    z[0] = ρz * z[-1] + σz * ϵz[x]
    g[0] = ρg * g[-1] + σg * ϵg[x]

    YGR[0] = gQ + 100 * (y[0] - y[-1] + z[0])
    INFL[0] = πA + 4 * π[0]
    FFR[0] = rA + piA + 4 * gQ + 4 * r[0]

    β = 1 / (1 + rA / 400)
end

# %% define parameters
MM.@parameters HS2017 begin

    rA = 0.5
    πA = 7.0
    gQ = 0.4
    τ = 2.0
    κ = 0.5
    ψ = 1.5
    ρR = 0.5
    ρg = 0.5
    ρz = 0.5
    σR = 0.4
    σg = 1.0
    σz = 0.5

    β = 0.99875

end

# %%
#============================SET UP TURING (NUTS) SAMPLER=============================#
# %% Setup prior distributions array
# import MacroModelling: Beta, Normal, InverseGamma

# %% define prior dists
# MacroModelling instead of Distributions because parameter distribution moments we have need transforming.
# Distributions expects α and β for Beta dists. We have μ and σ. MacroModelling.Beta transforms them, etc...
# Must import Turing first otherwise MacroModelling distribution exports won't work apparently...

# %% Specify prior distributions
import Turing
import MacroModelling: Beta, Normal, InverseGamma

prior_dists = [
    Gamma(0.50, 0.50, μσ=true),                  # rA   (r^(A))
    Gamma(7.00, 2.00, μσ=true),                  # piA  (π^(A))
    Normal(0.40, 0.20),                          # gQ   (γ^(Q))
    Gamma(2.00, 0.50, μσ=true),                  # τ
    Uniform(0.00, 1.00),                         # κ
    Gamma(1.50, 0.25, μσ=true),                  # ψ1
    Gamma(0.50, 0.25, μσ=true),                  # ψ2
    Uniform(0.00, 1.00),                         # ρR
    Uniform(0.00, 1.00),                         # ρg
    Uniform(0.00, 1.00),                         # ρz
    InverseGamma(0.501326, 0.262055, μσ=true),   # σR
    InverseGamma(1.253314, 0.655136, μσ=true),   # σg
    InverseGamma(0.626657, 0.327568, μσ=true)    # σz
];

# %%
#============================SET UP TURING AND PIGEONS SAMPLER=============================#
# %% Sampler imports
import DynamicPPL as DPPL

# %% Define PT-compatible Sampler function
Turing.@model function HS2017_loglik_func(prior_dists, data, m, on_loglik_failure; verbose=false)
    parameters ~ Turing.arraydist(prior_dists)

    if DPPL.leafcontext(__context__) !== DPPL.PriorContext() # Just following Pigeons.jl documentation for custom loglik
        llh = MM.get_loglikelihood(m, data, parameters, on_failure_loglikelihood=on_loglik_failure)

        if verbose
            @info "Loglikelihood: $llh and prior llh: $(Turing.logpdf(Turing.arraydist(prior_dists), parameters)) with params $parameters"
        end

        Turing.@addlogprob! llh
    end
end

failure_pen = -1e10
HS2017_loglik_pop = HS2017_loglik_func(prior_dists, data, HS2017, failure_pen)

# %%
#============================PIGEONS INITIALIZATION=============================#
# %% imports
using Random
import Pigeons

# %% Pigeons logpotential
init_params = HS2017.parameter_values
HS2017_PT_lp = Pigeons.TuringLogPotential(HS2017_loglik_pop)
typeof_HS2017_PT_LP = typeof(HS2017_PT_lp)

# %% Pigeons initialization
function Pigeons.initialization(target::typeof_HS2017_PT_LP, rng::AbstractRNG, _::Int64)
    result = DPPL.VarInfo(rng, target.model, DPPL.SampleFromPrior(), DPPL.PriorContext())

    result = DPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end
