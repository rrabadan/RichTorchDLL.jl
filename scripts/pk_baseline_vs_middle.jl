using ArgParse
using CairoMakie
using DataFrames
using RichTorchDLL
using UnROOT

CairoMakie.activate!(type = "png")  # Use PNG backend for saving figures

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--luminosity"
        help = "Peak luminosity (Default or Medium)"
        arg_type = String
        default = "Medium"

        "data-dir"
        help = "Directory containing input ROOT files"
        arg_type = String
        default = "./data"

        "--output-dir"
        help = "Output directory for saving plots"
        arg_type = String
        default = "figures"

        "--no-central-modules"
        help = "Whether to exclude central modules from the analysis"
        action = :store_true
    end
    return ArgParse.parse_args(s)
end

function load_data(args)

    luminosity = args["luminosity"]
    nocentralmod = args["no-central-modules"]

    filename_base = "Expert-ProtoTuple-Run5-$(luminosity)-baseline.root"
    if nocentralmod
        filename_base = "Expert-ProtoTuple-Run5-$(luminosity)-baseline-nocentralmod.root"
    end
    filepath_base = joinpath(args["data-dir"], filename_base)

    filename_mid = "Expert-ProtoTuple-Run5-$(luminosity)-middle.root"
    filepath_mid = joinpath(args["data-dir"], filename_mid)

    if !isfile(filepath_base)
        error("File not found: $filepath_base")
    end
    if !isfile(filepath_mid)
        error("File not found: $filepath_mid")
    end

    f_base = ROOTFile(filepath_base)
    f_mid = ROOTFile(filepath_mid)

    # Define which branches to read
    branches = [
        "RichDLLk",
        "RichDLLp",
        "TorchDLLk",
        "TorchDLLp",
        "MCParticleType",         # True particle ID
        "TrackP",                  # Momentum
    ]

    # Read tree
    t_base = LazyTree(f_base, "ChargedProtoTuple/protoPtuple", branches)
    df_base = DataFrame(t_base)
    println(nrow(df_base), " entries loaded from ", filepath_base)

    t_mid = LazyTree(f_mid, "ChargedProtoTuple/protoPtuple", branches)
    df_mid = DataFrame(t_mid)
    println(nrow(df_mid), " entries loaded from ", filepath_mid)

    return df_base, df_mid
end

args = parse_args()

if !(args["luminosity"] in ["Default", "Medium"])
    error(
        "Invalid peak luminosity scenario: $(args["luminosity"]). Valid options are: Default, Medium.",
    )
end

# Figure subdirectory
fig_subdir = "$(args["luminosity"])/proton-kaon"
if args["no-central-modules"]
    fig_subdir = "$(args["luminosity"])/nocentralmod/proton-kaon"
end
luminosity_text = L"L=1×10^{34} cm^{-2}s^{-1}"
if args["luminosity"] == "Default"
    luminosity_text = L"L=1.5×10^{34} cm^{-2}s^{-1}"
end

# Load data
try
    println("Loading data...")
    global df_base, df_middle = load_data(args)
catch e
    println("Cannot load data.")
    exit()
end


# create the per-subset DataFrames (filtered, rich, torch, valid) similar to
# `kpi_performance.jl`.
println("baseline dataset")
println()
datasets_base = prepare_dataset(
    df_base,
    particle_types = [is_proton, is_kaon],
    min_p = 2000,
    max_p = 20000,
    min_dll = -300,
    max_dll = 300,
    dlls = ["DLLk", "DLLp"],
)
println("middle dataset")
println()
datasets_middle = prepare_dataset(
    df_middle,
    particle_types = [is_proton, is_kaon],
    min_p = 2000,
    max_p = 20000,
    min_dll = -300,
    max_dll = 300,
    dlls = ["DLLk", "DLLp"],
)
println()

df_pk_base = datasets_base.filtered
rich_base = datasets_base.rich
torch_base = datasets_base.torch

df_pk_middle = datasets_middle.filtered
rich_middle = datasets_middle.rich
torch_middle = datasets_middle.torch

# Create binary labels (1 for kaons, 0 for pions)
labels_base = create_binary_labels(df_pk_base, is_proton)
rich_labels_base = create_binary_labels(rich_base, is_proton)
torch_labels_base = create_binary_labels(torch_base, is_proton)

labels_middle = create_binary_labels(df_pk_middle, is_proton)
rich_labels_middle = create_binary_labels(rich_middle, is_proton)
torch_labels_middle = create_binary_labels(torch_middle, is_proton)

println(nrow(torch_base), " entries with valid TORCH DLLs (baseline)")
println(nrow(torch_middle), " entries with valid TORCH DLLs (middle)")

# Define momentum bins with custom edges
momentum_bins = [
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
]
misid_rate = 0.05
yaxis_title_effcomp = L"p(\bar{p}) \text{ efficiency for 5% } K^{\pm} \text{ misID rate}"

torch_dll_base = torch_base.TorchDLLp - torch_base.TorchDLLk
torch_momentum_base = torch_base.TrackP ./ 1000  # Convert to GeV

torch_eff_base = efficiency_per_momentum_bin_at_misid_rate(
    torch_dll_base,
    torch_labels_base,
    torch_momentum_base,
    misid_rate,
    momentum_bins,
)

torch_dll_middle = torch_middle.TorchDLLp - torch_middle.TorchDLLk
torch_momentum_middle = torch_middle.TrackP ./ 1000  # Convert to GeV

torch_eff_middle = efficiency_per_momentum_bin_at_misid_rate(
    torch_dll_middle,
    torch_labels_middle,
    torch_momentum_middle,
    misid_rate,
    momentum_bins,
)

bin_centers_list = [torch_eff_base.bin_centers, torch_eff_middle.bin_centers]
bin_eff_list = [torch_eff_base.efficiency, torch_eff_middle.efficiency]
bin_efferr_list = [torch_eff_base.efficiency_error, torch_eff_middle.efficiency_error]

comparison = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = ["Baseline", "Middle"],
    title = "TORCH Proton Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:crimson :crimson],
    linestyles = [:dash :solid],
    legend_position = :lb,
    luminosity = luminosity_text,
)

save_figure(
    comparison.figure,
    "$(fig_subdir)/torch_efficiency_base-middle",
    figdir = args["output-dir"],
)

rich_dll_base = rich_base.RichDLLp - rich_base.RichDLLk
rich_momentum_base = rich_base.TrackP ./ 1000  # Convert to GeV

rich_eff_base = efficiency_per_momentum_bin_at_misid_rate(
    rich_dll_base,
    rich_labels_base,
    rich_momentum_base,
    misid_rate,
    momentum_bins,
)

rich_dll_middle = rich_middle.RichDLLp - rich_middle.RichDLLk
rich_momentum_middle = rich_middle.TrackP ./ 1000  # Convert to GeV

rich_eff_middle = efficiency_per_momentum_bin_at_misid_rate(
    rich_dll_middle,
    rich_labels_middle,
    rich_momentum_middle,
    misid_rate,
    momentum_bins,
)

bin_centers_list = [rich_eff_base.bin_centers, rich_eff_middle.bin_centers]
bin_eff_list = [rich_eff_base.efficiency, rich_eff_middle.efficiency]
bin_efferr_list = [rich_eff_base.efficiency_error, rich_eff_middle.efficiency_error]

rich_comparison = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = ["Baseline", "Middle"],
    title = "RICH Proton Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:royalblue :royalblue],
    linestyles = [:dash :solid],
    legend_position = :rt,
    luminosity = luminosity_text,
)

save_figure(
    rich_comparison.figure,
    "$(fig_subdir)/rich_efficiency_base-middle",
    figdir = args["output-dir"],
)

println("Optimizing combination model...")

dll_rich_base = df_pk_base.RichDLLp - df_pk_base.RichDLLk
dll_torch_base = df_pk_base.TorchDLLp - df_pk_base.TorchDLLk

dll_rich_middle = df_pk_middle.RichDLLp - df_pk_middle.RichDLLk
dll_torch_middle = df_pk_middle.TorchDLLp - df_pk_middle.TorchDLLk

scan_results_base = run_parameter_scan_1d(
    dll_rich_base,
    dll_torch_base,
    labels_base,
    :w,                 # scan_var
    -20.0:0.5:20.0,     # scan_range
    0.0,                # bias fixed_value
)
scan_results_middle = run_parameter_scan_1d(
    dll_rich_middle,
    dll_torch_middle,
    labels_middle,
    :w,                 # scan_var
    -20.0:0.5:20.0,     # scan_range
    0.0,                # bias fixed_value
)

println("Scans complete.")
println("Baseline Best weight found: w = $(scan_results_base.results.best)")
println("Middle Best weight found: w = $(scan_results_middle.results.best)")

# Get best weight from scan
base_best_w = scan_results_base.results.best.param
base_best_b = 0.0  # Using fixed bias of 0

middle_best_w = scan_results_middle.results.best.param
middle_best_b = 0.0  # Using fixed bias of 0


# Create combined scores
combined_dll_base = dll_rich_base .+ (base_best_w .* dll_torch_base .+ base_best_b)
combined_dll_middle =
    dll_rich_middle .+ (middle_best_w .* dll_torch_middle .+ middle_best_b)

momentum_base = df_pk_base.TrackP ./ 1000  # Convert to GeV
momentum_middle = df_pk_middle.TrackP ./ 1000  # Convert to GeV

rich_eff_base = efficiency_per_momentum_bin_at_misid_rate(
    dll_rich_base,
    labels_base,
    momentum_base,
    misid_rate,
    momentum_bins,
)

rich_eff_middle = efficiency_per_momentum_bin_at_misid_rate(
    dll_rich_middle,
    labels_middle,
    momentum_middle,
    misid_rate,
    momentum_bins,
)

comb_eff_base = efficiency_per_momentum_bin_at_misid_rate(
    combined_dll_base,
    labels_base,
    momentum_base,
    misid_rate,
    momentum_bins,
)

comb_eff_middle = efficiency_per_momentum_bin_at_misid_rate(
    combined_dll_middle,
    labels_middle,
    momentum_middle,
    misid_rate,
    momentum_bins,
)

_bin_centers_list = [
    rich_eff_base.bin_centers,
    rich_eff_middle.bin_centers,
    comb_eff_base.bin_centers,
    comb_eff_middle.bin_centers,
]
_bin_eff_list = [
    rich_eff_base.efficiency,
    rich_eff_middle.efficiency,
    comb_eff_base.efficiency,
    comb_eff_middle.efficiency,
]
_bin_efferr_list = [
    rich_eff_base.efficiency_error,
    rich_eff_middle.efficiency_error,
    comb_eff_base.efficiency_error,
    comb_eff_middle.efficiency_error,
]

comparison = compare_bin_efficiency_data(
    _bin_centers_list,
    _bin_eff_list,
    _bin_efferr_list,
    momentum_bins;
    figsize = (600, 500),
    labels = ["RICH Baseline", "RICH Middle", "Combined Baseline", "Combined Middle"],
    title = "Proton Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:royalblue, :royalblue, :black, :black],
    linestyles = [:dash, :solid, :dash, :solid],
    legend_position = :rt,
    luminosity = luminosity_text,
)

save_figure(
    comparison.figure,
    "$(fig_subdir)/comb_efficiency_base-middle",
    figdir = args["output-dir"],
)

all_scores = [dll_rich_base, dll_rich_middle, combined_dll_base, combined_dll_middle]
all_labels = [labels_base, labels_middle, labels_base, labels_middle]

curves_lowmom, curves_lowmom_log = compare_performance_curve(
    all_scores,
    all_labels,
    ["RICH baseline", "RICH middle", "RICH+TORCH baseline", "RICH+TORCH middle"],
    [:royalblue, :royalblue, :black, :black],
    linestyles = [:dash, :solid, :dash, :solid];
    figsize = (600, 500),
    title = " 2 < p < 20 GeV/c",
    xlabel = L"p(\bar{p}) \text{ efficiency}",
    ylabel = L"K^{\pm} \text{ missID rate}",
    luminosity = luminosity_text,
)

save_figure(
    curves_lowmom,
    "$(fig_subdir)/roc_lowmom_base-middle",
    figdir = args["output-dir"],
)
save_figure(
    curves_lowmom_log,
    "$(fig_subdir)/roc_lowmom_base-middle_log",
    figdir = args["output-dir"],
)
