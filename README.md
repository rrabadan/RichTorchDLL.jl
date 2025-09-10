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

## Running the RICH standalone performance plot

Once the project environment is instantiated, RICH standalone performance plot script can be run from the repository root. This will use the project environment and save figures into the `figures/` directory by default:

```bash
julia --project=. scripts/rich_standalone_performance.jl
```

## Data files and Git LFS

ROOT input files are tracked with Git LFS in this repository. When cloning the repo for the first time, ensure Git LFS is installed and fetch LFS objects so the data files are available to the scripts:

```bash
# install git-lfs (platform-specific)
git lfs install

# clone and pull LFS-managed files
git clone git@github.com:rrabadan/RichTorchDLL.jl.git
cd RichTorchDLL.jl
git lfs pull
```

If LFS objects are not pulled, the data directory will contain pointer files instead of real ROOT files and the scripts will fail when trying to open them.

