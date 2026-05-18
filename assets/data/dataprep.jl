# %%

using Pkg
cd(joinpath(@__DIR__, "..", ".."))
Pkg.activate(".")

# %%

using DataFrames, CSV

# %% load data

data_q = CSV.read("assets/data/fsdat.csv", DataFrame, header=false);

# %%

series = zeros(193, 2)
series[:, 2] = data_q[:, 1]
series[:, 1] = 1000 * data_q[:, 2] ./ data_q[:, 3];

# %%

Y_obs = series[:, 1]
P_obs = series[:, 2]

series = series[2:193, :] ./ series[1:192, :];

# %%

gy_obs = series[:, 1]
gp_obs = series[:, 2];

# %%

dat = DataFrame(hcat(gy_obs, gp_obs), [:gy_obs, :gp_obs]);
CSV.write("assets/data/FS2000_data_og.csv", dat)

# %% load data
dat = read("assets/data/FS2000_data_og.csv", DataFrame)
dat = log.(dat);
DataFrames.rename!(dat, Symbol.("log_" .* names(dat)));
DataFrames.select!(dat, [:log_gp_obs, :log_gy_obs]);
write("assets/data/FS2000_data.csv", dat)
