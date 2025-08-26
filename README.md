# RichTorchDLL.jl

Small utilities to combine RICH and TORCH DLL scores for particle identification
and to evaluate PID performance.

## Quick start (Julia project)

All commands assume you are in the repository root. Use a Julia project environment so
the correct dependencies from `Project.toml` / `Manifest.toml` are used.

1. Start Julia with the project activated:

```bash
julia --project=.
```

2. From the Julia REPL instantiate the environment (install packages):

```julia
using Pkg
Pkg.instantiate()
```

3. Open a notebook or run code interactively:

```julia
using RichTorchDLL

# Example: prepare data and compute mis-id/eff curve
X, y = prepare_data(rich_dll_k, torch_dll_k, labels)
model = combination_model(init_w=[1.0, 0.1], init_b=[0.0])
scores = vec(model(X))
pts = misid_eff_dataframe(scores, y)
```

## Common tasks

- Scan `(w,b)` grid: see `src/scan.jl` (`parameter_scan`, `parameter_scan_1d`).
- Compute working points (thresholds) and mis-id/eff curves: see `src/pidperformance.jl` (`find_thresholds_for_misid`, `misid_eff_points`, `misid_eff_dataframe`).
