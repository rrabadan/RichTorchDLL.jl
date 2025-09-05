using ArgParse
using CairoMakie
using DataFrames
using RichTorchDLL

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

        "--plot-dlls"
        help = "Whether to plot DLL distributions"
        action = :store_true
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
    global df = load_data(args["data-dir"], args["luminosity"], args["scenario"])
catch e
    println("Cannot load data.")
    exit()
end

datasets = prepare_dataset(
    df,
    particle_types = [is_pion, is_kaon],
    min_p = 2000,
    max_p = 15000,
    min_dll = -100,
    max_dll = 100,
    dlls = ["DLLk"],
)

richtorch = datasets.filtered
vrich = datasets.rich
vtorch = datasets.torch
vrichtorch = datasets.all_valid

labels = create_binary_labels(richtorch, is_kaon)
vrich_labels = create_binary_labels(vrich, is_kaon)
vtorch_labels = create_binary_labels(vtorch, is_kaon)
vlabels = create_binary_labels(vrichtorch, is_kaon)

plot_dlls = args["plot-dlls"]

if plot_dlls
    println("Plotting DLL distributions...")
    # Plot DLL distributions for pions and kaons
    pions = filter(row -> is_pion(row.MCParticleType), richtorch)
    kaons = filter(row -> is_kaon(row.MCParticleType), richtorch)

    fig = Figure(size = (800, 400))
    ax_rich = multi_histogram!(
        fig[1, 1],
        (pions.RichDLLk, kaons.RichDLLk),
        labels = [L"\pi^{\pm}" L"K^{\pm}"],
        xlabel = "DLLk",
        title = "RICH DLLk",
        histtype = :bar,
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

    # Plot DLL distributions for pions and kaons
    pions = filter(row -> is_pion(row.MCParticleType), vrichtorch)
    kaons = filter(row -> is_kaon(row.MCParticleType), vrichtorch)

    fig = Figure(size = (800, 400))
    ax_rich = multi_histogram!(
        fig[1, 1],
        (pions.RichDLLk, kaons.RichDLLk),
        labels = [L"\pi^{\pm}" L"K^{\pm}"],
        xlabel = "DLLk",
        title = "RICH DLLk",
        histtype = :bar,
    )
    ax_torch = multi_histogram!(
        fig[1, 2],
        (pions.TorchDLLk, kaons.TorchDLLk),
        labels = [L"\pi^{\pm}" L"K^{\pm}"],
        xlabel = "DLLk",
        title = "TORCH DLLk",
        histtype = :bar,
    )
    save_figure(fig, "$(fig_subdir)/dll_distributions_valid", figdir = args["output-dir"])
end

# Get DLL values and momentum
rich_dllk = richtorch.RichDLLk
torch_dllk = richtorch.TorchDLLk
momentum = richtorch.TrackP ./ 1000  # Convert to GeV/c

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
vrich_dllk = vrich.RichDLLk
vrich_momentum = vrich.TrackP ./ 1000

eff_vrich = efficiency_vs_momentum_with_per_bin_misid(
    vrich_dllk,
    vrich_labels,
    vrich_momentum,
    target_misid,
    momentum_bins;
    title = "RICH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :royalblue,
    legend_position = :rc,
)

eff_rich = efficiency_vs_momentum_with_per_bin_misid(
    rich_dllk,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "RICH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :royalblue,
    legend_position = :rc,
)

valid_rich_fraction = plot_nonzero_fraction_histogram(
    rich_dllk,
    momentum,
    momentum_bins;
    title = "Fraction of valid RICH DLLk",
    color = :gray,
)

# Create efficiency plots for TORCH DLLk
vtorch_dllk = vtorch.TorchDLLk
vtorch_momentum = vtorch.TrackP ./ 1000

eff_vtorch = efficiency_vs_momentum_with_per_bin_misid(
    vtorch_dllk,
    vtorch_labels,
    vtorch_momentum,
    target_misid,
    momentum_bins;
    title = "TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :crimson,
    legend_position = :rc,
)

eff_torch = efficiency_vs_momentum_with_per_bin_misid(
    torch_dllk,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :crimson,
    legend_position = :rc,
)

valid_torch_fraction = plot_nonzero_fraction_histogram(
    torch_dllk,
    momentum,
    momentum_bins;
    title = "Fraction of valid TORCH DLLk",
    color = :gray,
)

# Create combined scores
combined_dllk = rich_dllk .+ (best_w .* torch_dllk .+ best_b)
momentum = richtorch.TrackP ./ 1000

eff_combined = efficiency_vs_momentum_with_per_bin_misid(
    combined_dllk,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "Combined Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :black,
    legend_position = :rc,
)

# Compare combined against RICH DLLk's
rich_bin_data = eff_rich.bin_data
comb_bin_data = eff_combined.bin_data

bin_centers_list = [rich_bin_data.bin_centers, comb_bin_data.bin_centers]
bin_eff_list = [rich_bin_data.efficiency, comb_bin_data.efficiency]
bin_efferr_list = [rich_bin_data.efficiency_error, comb_bin_data.efficiency_error]

eff_comparison = compare_bin_efficiency_data(
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

# Compare combined against RICH DLLk's for all valid tracks
combined_vdllk = vrichtorch.RichDLLk .+ (best_w .* vrichtorch.TorchDLLk .+ best_b)
vmomentum = vrichtorch.TrackP ./ 1000

eff_vcombined = efficiency_vs_momentum_with_per_bin_misid(
    combined_vdllk,
    vlabels,
    vmomentum,
    target_misid,
    momentum_bins;
    title = "Combined Efficiency",
    xlabel = "Momentum [GeV/c]",
    color = :black,
    legend_position = :rc,
)

combined_bin_data = eff_combined.bin_data
vcombined_bin_data = eff_vcombined.bin_data

eff_combined_comparison = compare_bin_efficiency_data(
    [combined_bin_data.bin_centers, vcombined_bin_data.bin_centers],
    [combined_bin_data.efficiency, vcombined_bin_data.efficiency],
    [combined_bin_data.efficiency_error, vcombined_bin_data.efficiency_error],
    momentum_bins;
    labels = ["All Tracks", "Tracks with valid DLL"],
    title = "Combined Kaon Efficiency (5% Pion Misid)",
    xlabel = "Momentum [GeV/c]",
    colors = [:black, :black],
    linestyles = [:solid, :dash],
    legend_position = :rb,
)

rich_bin_data = eff_vrich.bin_data
comb_bin_data = eff_vcombined.bin_data

bin_centers_list = [rich_bin_data.bin_centers, comb_bin_data.bin_centers]
bin_eff_list = [rich_bin_data.efficiency, comb_bin_data.efficiency]
bin_efferr_list = [rich_bin_data.efficiency_error, comb_bin_data.efficiency_error]

eff_vcomparison = compare_bin_efficiency_data(
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

all_scores = [rich_dllk, combined_dllk]
all_labels = [labels, labels]

println("Plotting performance curve...")
curves, curves_log = compare_performance_curve(
    all_scores,
    all_labels,
    ["RICH", "RICH+TORCH"],
    [:royalblue, :black],
)

all_vscores = [vrichtorch.RichDLLk, combined_vdllk]
all_vlabels = [vlabels, vlabels]

println("Plotting performance curve...")
vcurves, vcurves_log = compare_performance_curve(
    all_vscores,
    all_vlabels,
    ["RICH", "RICH+TORCH"],
    [:royalblue, :black],
)

# Save figures
save_figure(eff_rich.figure, "$(fig_subdir)/efficiency_rich", figdir = args["output-dir"])
save_figure(
    eff_vrich.figure,
    "$(fig_subdir)/efficiency_rich_valid",
    figdir = args["output-dir"],
)
save_figure(
    valid_rich_fraction.figure,
    "$(fig_subdir)/valid_rich_fraction",
    figdir = args["output-dir"],
)
save_figure(eff_torch.figure, "$(fig_subdir)/efficiency_torch", figdir = args["output-dir"])
save_figure(
    eff_vtorch.figure,
    "$(fig_subdir)/efficiency_torch_valid",
    figdir = args["output-dir"],
)
save_figure(
    valid_torch_fraction.figure,
    "$(fig_subdir)/valid_torch_fraction",
    figdir = args["output-dir"],
)
save_figure(
    eff_combined.figure,
    "$(fig_subdir)/efficiency_combined",
    figdir = args["output-dir"],
)
save_figure(
    eff_vcombined.figure,
    "$(fig_subdir)/efficiency_combined_valid",
    figdir = args["output-dir"],
)
save_figure(
    eff_comparison.figure,
    "$(fig_subdir)/efficiency_comparison",
    figdir = args["output-dir"],
)
save_figure(
    eff_vcomparison.figure,
    "$(fig_subdir)/efficiency_comparison_valid",
    figdir = args["output-dir"],
)
save_figure(
    eff_vcombined.figure,
    "$(fig_subdir)/efficiency_combined_valid",
    figdir = args["output-dir"],
)
save_figure(
    eff_vcomparison.figure,
    "$(fig_subdir)/efficiency_comparison_valid",
    figdir = args["output-dir"],
)
save_figure(
    eff_combined_comparison.figure,
    "$(fig_subdir)/efficiency_combined_comparison",
    figdir = args["output-dir"],
)
save_figure(curves, "$(fig_subdir)/performance_curve", figdir = args["output-dir"])
save_figure(curves_log, "$(fig_subdir)/performance_curve_log", figdir = args["output-dir"])

save_figure(vcurves, "$(fig_subdir)/performance_curve_valid", figdir = args["output-dir"])
save_figure(
    vcurves_log,
    "$(fig_subdir)/performance_curve_log_valid",
    figdir = args["output-dir"],
)

println("Plots saved to $(args["output-dir"])/$(fig_subdir)/ directory")


# Print performance summary
#println("\nPerformance Summary (5% Misid Rate):")
#println("RICH threshold: $(round(result_rich.workingpoint.threshold, digits=2))")
#println("TORCH threshold: $(round(result_torch.workingpoint.threshold, digits=2))")
#println("Combined threshold: $(round(result_combined.workingpoint.threshold, digits=2))")
