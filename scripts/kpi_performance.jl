using ArgParse
using CairoMakie
using DataFrames
using RichTorchDLL
using UnROOT

CairoMakie.activate!(type = "png")  # Use PNG backend for saving figures

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--scenario"
        help = "Scenario name (baseline or middle)"
        arg_type = String
        default = "middle"

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
    end
    return ArgParse.parse_args(s)
end

function save_figure(
    fig,
    filename::String;
    figdir::String = "./figures",
    ext::String = "png",
)
    figpath = joinpath(figdir, "$(filename).$(ext)")
    # ensure directory exists (including any subdirectories in filename)
    mkpath(dirname(figpath))
    # Makie's save expects (filename, figure)
    save(figpath, fig)
    println("Figure saved to: $figpath")
end

function load_data(args)

    luminosity = args["luminosity"]
    scenario = args["scenario"]

    filename = "Expert-ProtoTuple-Run5-$(luminosity)-$(scenario).root"
    filepath = joinpath(args["data-dir"], filename)

    if !isfile(filepath)
        error("File not found: $filepath")
    end

    f = ROOTFile(filepath)

    # Define which branches to read
    branches = [
        "RichDLLk",
        "TorchDLLk",  # DLL variables
        "MCParticleType",         # True particle ID
        "TrackP",                  # Momentum
    ]

    # Read tree
    t = LazyTree(f, "ChargedProtoTuple/protoPtuple", branches)
    df = DataFrame(t)
    println(nrow(df), " entries loaded from ", filepath)

    return df
end

args = parse_args()

# Assert that args["scenario"] is either "baseline" or "medium"
if !(args["scenario"] in ["baseline", "middle"])
    error("Invalid scenario: $(args["scenario"]). Valid options are: baseline, medium.")
end

# Assert that args["scenario"] is either "Default" or "Medium"
if !(args["luminosity"] in ["Default", "Medium"])
    error(
        "Invalid peak luminosity scenario: $(args["luminosity"]). Valid options are: Default, Medium.",
    )
end

# Figure subdirectory
fig_subdir = "$(args["luminosity"])-$(args["scenario"])/kaon-pion"
savefig_func =
    (fig, filename, kwargs...) ->
        save_figure(fig, filename; figdir = "$(args["output-dir"])/$fig_subdir", kwargs...)

# Load data
try
    println("Loading data...")
    global df = load_data(args)
catch e
    println("Cannot load data.")
    exit()
end

# Prepare labels (1 for kaons, 0 for pions)
is_kaon(id) = abs(id) == 321
is_pion(id) = abs(id) == 211

# Filter for kaons and pions
df_kpi = filter(row -> is_kaon(row.MCParticleType) || is_pion(row.MCParticleType), df)
println("$(nrow(df_kpi)) entries after filtering for kaons and pions")

# Filter for momentum and DLLk
filter!(:TrackP => p -> 2000 < p < 15000, df_kpi);
println(
    "$(nrow(df_kpi)) entries after filtering for tracks with momentum 2 GeV < p < 15 GeV",
)

filter!([:RichDLLk, :TorchDLLk] => (r, t) -> -100 < r < 100 && -100 < t < 100, df_kpi);
println("$(nrow(df_kpi)) entries after filtering for DLLk in [-100, 100]")

df_torch = filter([:TorchDLLk] => t -> t != 0.0, df_kpi);
println("$(nrow(df_torch)) entries after filtering for valid TORCH DLLk")

df_rich = filter([:RichDLLk] => r -> r != 0.0, df_kpi);
println("$(nrow(df_rich)) entries after filtering for valid RICH DLLk")

# Create binary labels (1 for kaons, 0 for pions)
labels = Int.(is_kaon.(df_kpi.MCParticleType))
df_rich_labels = Int.(is_kaon.(df_rich.MCParticleType))
df_torch_labels = Int.(is_kaon.(df_torch.MCParticleType))

# Plot DLL distributions for pions and kaons
pions = filter(row -> is_pion(row.MCParticleType), df_kpi)
kaons = filter(row -> is_kaon(row.MCParticleType), df_kpi)

fig = Figure(size = (800, 400))
ax_rich = multi_histogram!(
    fig[1, 1],
    (pions.RichDLLk, kaons.RichDLLk),
    labels = [L"\pi^{\pm}" L"K^{\pm}"],
    xlabel = "DLLk",
    title = "RICH DLLk",
    histtype = :bar,
    #size=(800, 600)
)
ax_torch = multi_histogram!(
    fig[1, 2],
    (pions.TorchDLLk, kaons.TorchDLLk),
    labels = [L"\pi^{\pm}" L"K^{\pm}"],
    xlabel = "DLLk",
    title = "TORCH DLLk",
    histtype = :bar,
    #size=(800, 600)
)
save_figure(fig, "$(fig_subdir)/dll_distributions", figdir = args["output-dir"])

# Get DLL values and momentum
rich_dllk = df_kpi.RichDLLk
torch_dllk = df_kpi.TorchDLLk
momentum = df_kpi.TrackP

# Find optimal weight for combining RICH and TORCH
println("Optimizing combination model...")
scan_results = run_parameter_scan_1d(
    rich_dllk,
    torch_dllk,
    labels,
    :w,                 # scan_var
    -20.0:0.5:20.0,     # scan_range
    0.0,                # bias fixed_value
)
println("Scan complete.")
#println("Mean weight from repeated scans: ", scan_results.mean, " ± ", scan_results.std)
println("Best weight found: w = $(scan_results.results.best)")

# Save figure
save_figure(scan_results.figure, "$(fig_subdir)/scan_w", figdir = args["output-dir"])

# Get best weight from scan
best_w = scan_results.results.best.param
best_b = 0.0  # Using fixed bias of 0

# Plot efficiency vs momentum for RICH, TORCH, and combined classifier
println("Plotting efficiency vs momentum...")

# Define momentum bins (in GeV/c)
momentum_bins = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

# Find threshold for 5% misid rate
target_misid = 0.05

# Create efficiency plots for RICH DLLk
df_rich_dllk = df_rich.RichDLLk
df_rich_momentum = df_rich.TrackP ./ 1000

#result_rich = efficiency_vs_momentum_for_misid_rate(
result_rich = efficiency_vs_momentum_with_per_bin_misid(
    df_rich_dllk,
    df_rich_labels,
    df_rich_momentum,
    target_misid,
    momentum_bins;
    title = "RICH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :royalblue,
    legend_position = :rc,
)

# Create efficiency plots for RICH DLLk
df_torch_dllk = df_torch.TorchDLLk
df_torch_momentum = df_torch.TrackP ./ 1000

#result_torch = efficiency_vs_momentum_for_misid_rate(
result_torch = efficiency_vs_momentum_with_per_bin_misid(
    df_torch_dllk,
    df_torch_labels,
    df_torch_momentum,
    target_misid,
    momentum_bins;
    title = "TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :crimson,
    legend_position = :rc,
)

# Create combined scores
combined_dllk = rich_dllk .+ (best_w .* torch_dllk .+ best_b)

# Convert momentum to GeV/c for plotting
momentum_gev = momentum ./ 1000

result_combined = efficiency_vs_momentum_with_per_bin_misid(
    combined_dllk,
    labels,
    momentum_gev,
    target_misid,
    momentum_bins;
    title = "RICH+TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :black,
    legend_position = :rc,
)


# Compare combined against RICH DLLk's
rich_bin_data = result_rich.bin_data
comb_bin_data = result_combined.bin_data

bin_centers_list = [rich_bin_data.bin_centers, comb_bin_data.bin_centers]
bin_eff_list = [rich_bin_data.efficiency, comb_bin_data.efficiency]
bin_efferr_list = [rich_bin_data.efficiency_error, comb_bin_data.efficiency_error]

all_scores = [df_rich_dllk, combined_dllk]
all_labels = [df_rich_labels, labels]
# all_momentum = [df_rich_momentum, momentum_gev]
# Get thresholds for each DLLk at 5% misid rate
# thresholds = [result_rich.workingpoint.threshold, result_combined.workingpoint.threshold]

comparison = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = ["RICH", "RICH+TORCH"],
    title = "Kaon Efficiency (5% Pion Misid)",
    xlabel = "Momentum [GeV/c]",
    colors = [:royalblue, :black],
    legend_position = :rb,
)

println("Plotting performance curve...")
curves, curves_log = compare_performance_curve(
    all_scores,
    all_labels,
    ["RICH", "RICH+TORCH"],
    [:royalblue, :black],
)

# Save figures
save_figure(
    result_rich.figure,
    "$(fig_subdir)/efficiency_rich",
    figdir = args["output-dir"],
)
save_figure(
    result_torch.figure,
    "$(fig_subdir)/efficiency_torch",
    figdir = args["output-dir"],
)
save_figure(
    result_combined.figure,
    "$(fig_subdir)/efficiency_combined",
    figdir = args["output-dir"],
)
save_figure(
    comparison.figure,
    "$(fig_subdir)/efficiency_comparison",
    figdir = args["output-dir"],
)
save_figure(curves, "$(fig_subdir)/performance_curve", figdir = args["output-dir"])
save_figure(curves_log, "$(fig_subdir)/performance_curve_log", figdir = args["output-dir"])

println("Plots saved to $(args["output-dir"])/$(fig_subdir)/ directory")


# Print performance summary
#println("\nPerformance Summary (5% Misid Rate):")
#println("RICH threshold: $(round(result_rich.workingpoint.threshold, digits=2))")
#println("TORCH threshold: $(round(result_torch.workingpoint.threshold, digits=2))")
#println("Combined threshold: $(round(result_combined.workingpoint.threshold, digits=2))")
