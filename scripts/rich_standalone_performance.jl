using ArgParse
using CairoMakie
using DataFrames
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
    end
    return ArgParse.parse_args(s)
end

args = parse_args()

# Figure subdirectory
fig_subdir = "RichSA/kaon-pion"
luminosity_base_text = L"\text{Baseline, } L=1.5×10^{34} cm^{-2}s^{-1}"
luminosity_midd_text = L"\text{Middle, } L=1.0×10^{34} cm^{-2}s^{-1}"

# Load data
try
    println("Loading data...")
    global df_base = load_data(args["data-dir"], "Default", "baseline", false)
    global df_midd = load_data(args["data-dir"], "Medium", "middle", false)
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
    particle_types = [is_kaon, is_pion],
    min_p = 3000,
    max_p = 200000,
    min_pt = 500,
    max_pt = 100000,
    min_dll = -1000,
    max_dll = 1000,
    dlls = ["DLLk"],
)
println("middle dataset")
println()
datasets_middle = prepare_dataset(
    df_midd,
    particle_types = [is_kaon, is_pion],
    min_p = 3000,
    max_p = 200000,
    min_pt = 400,
    max_pt = 100000,
    min_dll = -1000,
    max_dll = 1000,
    dlls = ["DLLk"],
)
println()

rich_base = datasets_base.rich

rich_middle = datasets_middle.rich

# Create binary labels (1 for kaons, 0 for pions)
rich_labels_base = create_binary_labels(rich_base, is_kaon)

rich_labels_middle = create_binary_labels(rich_middle, is_kaon)

# Define momentum bins with custom edges
momentum_bins = [
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
    15.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    150.0,
    200.0,
]
misid_rate = 0.01
yaxis_title_effcomp = L"K^{\pm} \text{ efficiency for 1% } \pi^{\pm} \text{ misID rate}"

rich_dllk_base = rich_base.RichDLLk
rich_momentum_base = rich_base.TrackP ./ 1000  # Convert to GeV

rich_eff_base = efficiency_per_momentum_bin_at_misid_rate(
    rich_dllk_base,
    rich_labels_base,
    rich_momentum_base,
    misid_rate,
    momentum_bins,
)

rich_dllk_middle = rich_middle.RichDLLk
rich_momentum_middle = rich_middle.TrackP ./ 1000  # Convert to GeV

rich_eff_middle = efficiency_per_momentum_bin_at_misid_rate(
    rich_dllk_middle,
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
    labels = [luminosity_base_text, luminosity_midd_text],
    title = "",
    xlabel = "Momentum [GeV/c]",
    ylabel = yaxis_title_effcomp,
    colors = [:royalblue :orange],
    linestyles = [:solid :solid],
    legend_position = :lb,
    show_bin_edges = false,
)

ax = rich_comparison.ax
# Log X
#ax.xscale = Makie.pseudolog10
ax.xscale = log10
ax.xgridvisible = true
ax.ygridvisible = true
ax.xminorticks = IntervalsBetween(10)
ax.xminorticksvisible = true
ax.xticks = [1, 10, 100]

save_figure(
    rich_comparison.figure,
    "$(fig_subdir)/rich_efficiency_base-middle",
    figdir = args["output-dir"],
)
