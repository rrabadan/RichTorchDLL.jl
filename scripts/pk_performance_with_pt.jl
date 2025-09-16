using ArgParse
using CairoMakie
using DataFrames
using LaTeXStrings
using RichTorchDLL
using UnROOT

CairoMakie.activate!(type = "png")  # Use PNG backend for saving figures

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "data-dir"
        help = "Directory containing input ROOT files"
        arg_type = String
        default = "./data"

        "--output-dir"
        help = "Output directory for saving plots"
        arg_type = String
        default = "figures"

        "--pt-cuts"
        help = "Comma-separated list of pT cuts in MeV (e.g. '0,300,500'). Use 0 for no cut."
        arg_type = String
        default = "0, 300, 500"
    end
    return ArgParse.parse_args(s)
end

args = parse_args()

# Figure subdirectory
fig_subdir = "WithPt/proton-kaon"
luminosity_midd_text = L"\text{Middle, } L=1.0×10^{34} cm^{-2}s^{-1}"

# Load data
try
    println("Loading data...")
    global df_midd = load_data(args["data-dir"], "Medium", "middle", false)
catch e
    println("Cannot load data.")
    exit()
end


# create the per-subset DataFrames (filtered, rich, torch, valid) similar to
# `kpi_performance.jl`.
println()
datasets = prepare_dataset(
    df_midd,
    particle_types = [is_proton, is_kaon],
    min_p = 2000,
    max_p = 20000,
    min_pt = 0,
    max_pt = 100000,
    min_dll = -1000,
    max_dll = 1000,
    dlls = ["DLLk", "DLLp"],
)

richtorch = datasets.filtered
rich = datasets.rich
torch = datasets.torch

# Create binary labels (1 for kaons, 0 for pions)
richtorch_labels = create_binary_labels(richtorch, is_proton)
torch_labels = create_binary_labels(torch, is_proton)
rich_labels = create_binary_labels(rich, is_proton)

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

# Find threshold for 5% and 1% misid rate
misid_rate_torch = 0.01
misid_rate = 0.01

yaxis_title_effcomp_torch =
    L"p(\bar{p}) \text{ efficiency for 1% } K^{\pm} \text{ misID rate}"
yaxis_title_effcomp = L"p(\bar{p}) \text{ efficiency for 1% } K^{\pm} \text{ misID rate}"


# parse pt cuts from CLI (given in MeV), default "0,500" -> [0, 500]
pt_cut_arg = args["pt-cuts"]
pt_cuts_mev = parse.(Int, split(pt_cut_arg, ","))
pt_cuts = pt_cuts_mev ./ 1000.0  # convert to GeV


function make_mask_pt(array_pt, ptcut_geV)
    if ptcut_geV <= 0
        return trues(length(array_pt))
    else
        return array_pt .> ptcut_geV
    end
end

function label_for_pt(ptcut)
    if ptcut <= 0
        return "no pT cut"
    else
        s = "p_{T} > $(ptcut) \\text{ GeV/c}"
        return LaTeXString(s)
    end
end

function compute_efficiencies_for_cuts(
    dll_array,
    labels_array,
    momentum_array,
    pt_array,
    pt_cuts,
    misid,
    bins,
)
    effs = Vector{Any}()
    for pt in pt_cuts
        mask = make_mask_pt(pt_array, pt)
        push!(
            effs,
            efficiency_per_momentum_bin_at_misid_rate(
                dll_array[mask],
                labels_array[mask],
                momentum_array[mask],
                misid,
                bins,
            ),
        )
    end
    return effs
end


function compute_combined_efficiencies_for_cuts(
    rich_dll_array,
    torch_dll_array,
    labels_array,
    momentum_array,
    pt_array,
    pt_cuts,
    misid,
    bins,
)
    effs = Vector{Any}()
    for pt in pt_cuts
        mask = make_mask_pt(pt_array, pt)
        println("Scan for pT cut: ", pt, " GeV/c, number of tracks: ", sum(mask))
        scan_results = run_parameter_scan_1d(
            rich_dll_array[mask],
            torch_dll_array[mask],
            labels_array[mask],
            :w,                 # scan_var
            -20.0:0.5:20.0,     # scan_range
            0.0,                # bias fixed_value
        )
        #println("Mean weight from repeated scans: ", scan_results.mean, " ± ", scan_results.std)
        println("Best weight found: w = $(scan_results.results.best)")
        combined_dll =
            rich_dll_array[mask] .+
            (scan_results.results.best.param .* torch_dll_array[mask])
        push!(
            effs,
            efficiency_per_momentum_bin_at_misid_rate(
                combined_dll,
                labels_array[mask],
                momentum_array[mask],
                misid,
                bins,
            ),
        )
    end
    return effs
end

luminosity_text = L"L=1×10^{34} cm^{-2}s^{-1}"

# Build labels for legend and compute efficiencies for each dataset
pt_labels = [label_for_pt(p) for p in pt_cuts]

# TORCH: compute efficiencies for all cuts
torch_dll = torch.TorchDLLp - torch.TorchDLLk
torch_eff_list = compute_efficiencies_for_cuts(
    torch_dll,
    torch_labels,
    torch.TrackP ./ 1000,
    torch.TrackPt ./ 1000,
    pt_cuts,
    misid_rate_torch,
    momentum_bins,
)
bin_centers_list = [e.bin_centers for e in torch_eff_list]
bin_eff_list = [e.efficiency for e in torch_eff_list]
bin_efferr_list = [e.efficiency_error for e in torch_eff_list]

N = length(torch_eff_list)
torch_comp = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = pt_labels,
    title = "TORCH Standalone",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp_torch,
    colors = [:salmon2, :firebrick1, :crimson],
    linestyles = [:dot, :dash, :solid],
    legend_position = :cb,
    show_bin_edges = true,
    luminosity = luminosity_text,
)
save_figure(
    torch_comp.figure,
    "$(fig_subdir)/torch_ptcuts_comparison",
    figdir = args["output-dir"],
)

# RICH: compute efficiencies for all cuts (use rich's TrackPt)
rich_dll = rich.RichDLLp - rich.RichDLLk
rich_eff_list = compute_efficiencies_for_cuts(
    rich_dll,
    rich_labels,
    rich.TrackP ./ 1000,
    rich.TrackPt ./ 1000,
    pt_cuts,
    misid_rate,
    momentum_bins,
)
bin_centers_list = [e.bin_centers for e in rich_eff_list]
bin_eff_list = [e.efficiency for e in rich_eff_list]
bin_efferr_list = [e.efficiency_error for e in rich_eff_list]

N = length(rich_eff_list)
rich_comp = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = pt_labels,
    title = "RICH Standalone",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:skyblue, :slateblue, :royalblue],
    #linestyles=Iterators.cycle([:solid, :dash, :dot, :dashdot]) |> collect |> x -> x[1:N],
    linestyles = [:dot, :dash, :solid],
    legend_position = :lb,
    show_bin_edges = true,
    luminosity = luminosity_text,
)
save_figure(
    rich_comp.figure,
    "$(fig_subdir)/rich_ptcuts_comparison",
    figdir = args["output-dir"],
)

# Combined: compute efficiencies for all cuts (use richtorch TrackPt)
combined_eff_list = compute_combined_efficiencies_for_cuts(
    richtorch.RichDLLp - richtorch.RichDLLk,
    richtorch.TorchDLLp - richtorch.TorchDLLk,
    richtorch_labels,
    richtorch.TrackP ./ 1000,
    richtorch.TrackPt ./ 1000,
    pt_cuts,
    misid_rate,
    momentum_bins,
)
bin_centers_list = [e.bin_centers for e in combined_eff_list]
bin_eff_list = [e.efficiency for e in combined_eff_list]
bin_efferr_list = [e.efficiency_error for e in combined_eff_list]

N = length(combined_eff_list)
comb_comp = compare_bin_efficiency_data(
    bin_centers_list,
    bin_eff_list,
    bin_efferr_list,
    momentum_bins;
    labels = pt_labels,
    title = "RICH+TORCH",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:gray50, :gray30, :black],
    linestyles = [:dot, :dash, :solid],
    legend_position = :rt,
    show_bin_edges = true,
    luminosity = luminosity_text,
)
save_figure(
    comb_comp.figure,
    "$(fig_subdir)/combined_ptcuts_comparison",
    figdir = args["output-dir"],
)


richtorch_richdll = richtorch.RichDLLp - richtorch.RichDLLk
richtorch_eff_list = compute_efficiencies_for_cuts(
    richtorch_richdll,
    richtorch_labels,
    richtorch.TrackP ./ 1000,
    richtorch.TrackPt ./ 1000,
    pt_cuts,
    misid_rate,
    momentum_bins,
)

_bin_centers_list = [
    richtorch_eff_list[1].bin_centers,
    richtorch_eff_list[end].bin_centers,
    combined_eff_list[1].bin_centers,
    combined_eff_list[end].bin_centers,
]
_bin_eff_list = [
    richtorch_eff_list[1].efficiency,
    richtorch_eff_list[end].efficiency,
    combined_eff_list[1].efficiency,
    combined_eff_list[end].efficiency,
]
_bin_efferr_list = [
    richtorch_eff_list[1].efficiency_error,
    richtorch_eff_list[end].efficiency_error,
    combined_eff_list[1].efficiency_error,
    combined_eff_list[end].efficiency_error,
]

labels = [
    L"\text{RICH (no pT cut)}",
    L"\text{RICH} (p_{T} > 0.5 \text{ GeV/c})",
    L"\text{RICH+TORCH (no pT cut)}",
    L"\text{RICH+TORCH} (p_{T} > 0.5 \text{ GeV/c})",
]
comparison = compare_bin_efficiency_data(
    _bin_centers_list,
    _bin_eff_list,
    _bin_efferr_list,
    momentum_bins;
    labels = labels,
    title = "",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:royalblue, :royalblue, :black, :black],
    linestyles = [:dash, :solid, :dash, :solid],
    legend_position = :rt,
    #luminosity=luminosity_text,
)

save_figure(
    comparison.figure,
    "$(fig_subdir)/rich_torch_comb_efficiency_comparison",
    figdir = args["output-dir"],
)
