using ArgParse
using CairoMakie
using DataFrames
using LaTeXStrings
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

        "--no-central-modules"
        help = "Whether to exclude central modules from the analysis"
        action = :store_true
    end
    return ArgParse.parse_args(s)
end

args = parse_args()

# Validate inputs
if !(args["scenario"] in ["baseline", "middle"])
    error("Invalid scenario: $(args["scenario"]). Valid options are: baseline, medium.")
end
if !(args["luminosity"] in ["Default", "Medium"])
    error(
        "Invalid peak luminosity scenario: $(args["luminosity"]). Valid options are: Default, Medium.",
    )
end

no_central_modules = args["no-central-modules"]
if no_central_modules && (args["scenario"] == "middle")
    error("Central modules excluded in the middle scenario by default.")
    exit()
end

# Figure subdirectory
fig_subdir = "$(args["luminosity"])-$(args["scenario"])/proton-kaon"
if no_central_modules
    fig_subdir = "$(args["luminosity"])-$(args["scenario"])-nocentralmod/proton-kaon"
end
savefig_func =
    (fig, filename, kwargs...) ->
        save_figure(fig, filename; figdir = "$(args["output-dir"])/$fig_subdir", kwargs...)

# Luminosity text
luminosity_text = L"L=1×10^{34} cm^{-2}s^{-1}"
if args["luminosity"] == "Default"
    luminosity_text = L"L=1.5×10^{34} cm^{-2}s^{-1}"
end

# Load data
try
    println("Loading data...")
    global df = load_data(
        args["data-dir"],
        args["luminosity"],
        args["scenario"],
        no_central_modules,
    )
catch e
    println("Cannot load data: $e")
    exit()
end

datasets = prepare_dataset(
    df,
    particle_types = [is_proton, is_kaon],
    min_p = 2000,
    max_p = 20000,
    min_dll = -300,
    max_dll = 300,
    dlls = ["DLLk", "DLLp"],
)

richtorch = datasets.filtered
vrich = datasets.rich
vtorch = datasets.torch
vrichtorch = datasets.all_valid

labels = create_binary_labels(richtorch, is_proton)
vrich_labels = create_binary_labels(vrich, is_proton)
vtorch_labels = create_binary_labels(vtorch, is_proton)
vlabels = create_binary_labels(vrichtorch, is_proton)

plot_dlls = args["plot-dlls"]

if plot_dlls
    println("Plotting DLL distributions...")
    protons = filter(row -> is_proton(row.MCParticleType), richtorch)
    kaons = filter(row -> is_kaon(row.MCParticleType), richtorch)

    fig = Figure(size = (800, 400))
    ax_rich = multi_histogram!(
        fig[1, 1],
        (protons.RichDLLp - protons.RichDLLk, kaons.RichDLLp - kaons.RichDLLk),
        labels = [L"p" L"K"],
        xlabel = "DLLp - DLLk",
        title = "RICH DLLp - DLLk",
        histtype = :bar,
        limits = ((-100, 100), nothing),
    )
    ax_torch = multi_histogram!(
        fig[1, 2],
        (protons.TorchDLLp - protons.TorchDLLk, kaons.TorchDLLp - kaons.TorchDLLk),
        labels = [L"p" L"K"],
        xlabel = "DLLp - DLLk",
        title = "TORCH DLLp - DLLk",
        histtype = :bar,
        limits = ((-100, 100), nothing),
    )
    save_figure(fig, "$(fig_subdir)/dll_distributions", figdir = args["output-dir"])

    # valid DLLs
    protons = filter(row -> is_proton(row.MCParticleType), vrichtorch)
    kaons = filter(row -> is_kaon(row.MCParticleType), vrichtorch)
    fig = Figure(size = (800, 400))
    multi_histogram!(
        fig[1, 1],
        (protons.RichDLLp - protons.RichDLLk, kaons.RichDLLp - kaons.RichDLLk),
        labels = [L"p" L"K"],
        xlabel = "DLLp - DLLk",
        title = "RICH DLLp - DLLk",
        histtype = :bar,
        limits = ((-100, 100), nothing),
    )
    multi_histogram!(
        fig[1, 2],
        (protons.TorchDLLp - protons.TorchDLLk, kaons.TorchDLLp - kaons.TorchDLLk),
        labels = [L"p" L"K"],
        xlabel = "DLLp - DLLk",
        title = "TORCH DLLp - DLLk",
        histtype = :bar,
        limits = ((-100, 100), nothing),
    )
    save_figure(fig, "$(fig_subdir)/dll_distributions_valid", figdir = args["output-dir"])
end

# Compute DLL scores and momentum
rich_dll = richtorch.RichDLLp - richtorch.RichDLLk
torch_dll = richtorch.TorchDLLp - richtorch.TorchDLLk
momentum = richtorch.TrackP ./ 1000

println("Optimizing combination model...")
scan_results = run_parameter_scan_1d(rich_dll, torch_dll, labels, :w, -20.0:0.5:20.0, 0.0)
println("Best weight found: w = $(scan_results.results.best)")
save_figure(scan_results.figure, "$(fig_subdir)/scan_w", figdir = args["output-dir"])

best_w = scan_results.results.best.param
best_b = 0.0

println("Plotting efficiency vs momentum...")
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
target_misid = 0.05

yaxis_title_effmom = L"p(\bar{p}) \text{ efficiency (} K^{\pm} \text{ misID rate)}"


# RICH
vrich_dll = vrich.RichDLLp - vrich.RichDLLk
vrich_momentum = vrich.TrackP ./ 1000

eff_vrich = efficiency_vs_momentum_with_per_bin_misid(
    vrich_dll,
    vrich_labels,
    vrich_momentum,
    target_misid,
    momentum_bins;
    title = "RICH Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :royalblue,
    legend_position = :rt,
    luminosity = luminosity_text,
)

eff_rich = efficiency_vs_momentum_with_per_bin_misid(
    rich_dll,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "RICH Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :royalblue,
    legend_position = :rt,
    luminosity = luminosity_text,
)

valid_rich_fraction = plot_nonzero_fraction_histogram(
    rich_dll,
    momentum,
    momentum_bins;
    title = "Valid RICH DLL",
    color = :gray,
)
add_luminosity_text!(
    valid_rich_fraction.ax,
    args["luminosity"],
    position = :rt,
    fontsize = 14,
)

# TORCH
vtorch_dll = vtorch.TorchDLLp - vtorch.TorchDLLk
vtorch_momentum = vtorch.TrackP ./ 1000

eff_vtorch = efficiency_vs_momentum_with_per_bin_misid(
    vtorch_dll,
    vtorch_labels,
    vtorch_momentum,
    target_misid,
    momentum_bins;
    title = "TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :crimson,
    legend_position = :rt,
    luminosity = luminosity_text,
)

eff_torch = efficiency_vs_momentum_with_per_bin_misid(
    torch_dll,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "TORCH Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :crimson,
    legend_position = :rt,
    luminosity = luminosity_text,
)

valid_torch_fraction = plot_nonzero_fraction_histogram(
    torch_dll,
    momentum,
    momentum_bins;
    title = "Valid TORCH DLL",
    color = :gray,
)
add_luminosity_text!(
    valid_torch_fraction.ax,
    args["luminosity"],
    position = :rt,
    fontsize = 14,
)

# Combined
combined_dll = rich_dll .+ (best_w .* torch_dll .+ best_b)
momentum = richtorch.TrackP ./ 1000

eff_combined = efficiency_vs_momentum_with_per_bin_misid(
    combined_dll,
    labels,
    momentum,
    target_misid,
    momentum_bins;
    title = "Combined Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :black,
    legend_position = :rt,
    luminosity = luminosity_text,
)

# Comparisons
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
    title = "Proton Efficiency (5% Kaon Misid)",
    xlabel = "Momentum [GeV/c]",
    colors = [:royalblue, :black],
    legend_position = :rb,
    luminosity = luminosity_text,
)

# Valid combined
combined_vdll =
    vrichtorch.RichDLLp - vrichtorch.RichDLLk .+
    (best_w .* (vrichtorch.TorchDLLp - vrichtorch.TorchDLLk) .+ best_b)
vmomentum = vrichtorch.TrackP ./ 1000

eff_vcombined = efficiency_vs_momentum_with_per_bin_misid(
    combined_vdll,
    vlabels,
    vmomentum,
    target_misid,
    momentum_bins;
    title = "Combined Efficiency",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effmom,
    color = :black,
    legend_position = :lc,
    luminosity = luminosity_text,
)

combined_bin_data = eff_combined.bin_data
vcombined_bin_data = eff_vcombined.bin_data

eff_combined_comparison = compare_bin_efficiency_data(
    [combined_bin_data.bin_centers, vcombined_bin_data.bin_centers],
    [combined_bin_data.efficiency, vcombined_bin_data.efficiency],
    [combined_bin_data.efficiency_error, vcombined_bin_data.efficiency_error],
    momentum_bins;
    labels = ["All Tracks", "Tracks with valid DLL"],
    title = "Combined Proton Efficiency (5% Kaon Misid)",
    xlabel = "Momentum [GeV/c]",
    colors = [:black, :black],
    linestyles = [:solid, :dash],
    legend_position = :rb,
    luminosity = luminosity_text,
)

# more comparisons for valid tracks
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
    title = "Kaon Efficiency",
    xlabel = "Momentum [GeV/c]",
    colors = [:royalblue, :black],
    legend_position = :rb,
    luminosity = luminosity_text,
)

all_scores = [rich_dll, combined_dll]
all_labels = [labels, labels]

println("Plotting performance curve...")
curves, curves_log = compare_performance_curve(
    all_scores,
    all_labels,
    ["RICH", "RICH+TORCH"],
    [:royalblue, :black],
)

all_vscores = [vrichtorch.RichDLLp - vrichtorch.RichDLLk, combined_vdll]
all_vlabels = [vlabels, vlabels]

println("Plotting performance curve (valid)...")
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

println("Plots saved to $(args["output-dir"]) / $(fig_subdir) / directory")
