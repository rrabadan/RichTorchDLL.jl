using ArgParse
using DataFrames
using LaTeXStrings
using Plots
using StatsPlots
using UnROOT

using RichTorchDLL

function parse_args()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input-file"
        help = "Input ROOT file"
        arg_type = String
        required = true

        "--scenario"
        help = "Scenario name (baseline or medium)"
        arg_type = String
        default = "medium"

        "--output-dir"
        help = "Output directory for saving plots"
        arg_type = String
        default = "figures"
    end
    return ArgParse.parse_args(s)
end


function collect_data(args::Dict{String,Any})

    file = ROOTFile(args["input-file"])

    t = LazyTree(
        file,
        "ChargedProtoTuple/protoPtuple",
        ["TrackP", "MCParticleType", "RichDLLp", "TorchDLLp"],
    )

    df = DataFrame(t)

    pions_protons =
        filter(:MCParticleType => pdgid -> abs(pdgid) == 211 || abs(pdgid) == 2212, df)

    filter!(:TrackP => p -> p < 20000, pions_protons)
    filter!(
        [:RichDLLp, :TorchDLLp] => (r, t) -> -100 < r < 100 && -100 < t < 100,
        pions_protons,
    )
    select!(
        pions_protons,
        :,
        :MCParticleType => ByRow(pdgid -> (abs(pdgid) == 211 ? 0 : 1)) => :label,
    )

    return pions_protons
end

function plot_dll_distributions(pions_protons::DataFrame, args::Dict{String,Any})

    # Create two side-by-side histograms (TorchDLLp | RichDLLp) in a single figure
    pions = filter(:MCParticleType => pdgid -> abs(pdgid) == 211, pions_protons)
    protons = filter(:MCParticleType => pdgid -> abs(pdgid) == 2212, pions_protons)

    # Create a plot with appropriate margins to emulate matplotlib's tight_layout
    plt = plot(
        layout=(1, 2),
        size=(1000, 450),
        left_margin=10Plots.mm,
        right_margin=12Plots.mm,
        bottom_margin=8Plots.mm,
        top_margin=5Plots.mm,
        legend=:topright,
        dpi=300,
    )

    # Left: TorchDLLp
    histogram!(
        plt[1],
        pions.TorchDLLp;
        bins=100,
        label=L"\pi",
        xlabel="TorchDLLp",
        alpha=0.4,
        normalize=true,
        xlims=(-100, 100),
        title="TorchDLLp distribution",
    )
    histogram!(
        plt[1],
        protons.TorchDLLp;
        bins=100,
        label="p",
        alpha=0.5,
        normalize=true,
    )

    # Right: RichDLLp
    histogram!(
        plt[2],
        pions.RichDLLp;
        bins=100,
        label=L"\pi",
        xlabel="RichDLLp",
        alpha=0.4,
        normalize=true,
        xlims=(-100, 100),
        title="RichDLLp distribution",
    )
    histogram!(
        plt[2],
        protons.RichDLLp;
        bins=100,
        label="p",
        alpha=0.5,
        normalize=true,
    )

    outdir = args["output-dir"]
    scenario = args["scenario"]

    # Set figure title for the entire plot
    plot!(
        plt,
        plot_title="DLLp Distributions for $(scenario) scenario",
        plot_titlefontsize=12,
        titlefontsize=10,
    )

    # Save with high quality
    savefig(plt, joinpath(outdir, "run5_$(scenario)_dllp.png"))

    # Also save as PDF for publication quality
    # savefig(plt, joinpath(outdir, "run5_$(scenario)_dllp.pdf"))
end

function combination_scan_1d(
    pions_protons::DataFrame,
    range::Tuple{Float64,Float64,Float64},
    args::Dict{String,Any};
    suffix::String="",
    param::Symbol=:w
)
    rich_dll_p = pions_protons[:, :RichDLLp]
    torch_dll_p = pions_protons[:, :TorchDLLp]
    labels = pions_protons[:, :label]

    fixed_value = 0.0
    if param == :b
        fixed_value = 1.0
    end

    scan_results = parameter_scan_1d(
        rich_dll_p,
        torch_dll_p,
        labels,
        scan=param,
        fixed_value=fixed_value,
        scan_range=range[1]:range[2]:range[3],
    )

    # Visualize AUC
    p1 = visualize_1d_scan(scan_results, metric=:auc)

    # Visualize efficiency
    p2 = visualize_1d_scan(scan_results, metric=:efficiency)

    # Visualize misidentification probability
    p3 = visualize_1d_scan(scan_results, metric=:misid)

    # Visualize purity
    p4 = visualize_1d_scan(scan_results, metric=:purity)

    # Combine plots
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))

    outdir = args["output-dir"]
    scenario = args["scenario"]

    if !isempty(suffix)
        suffix = "_" * suffix
    end

    # Save with high quality
    savefig(plt, joinpath(outdir, "run5-$(scenario)_combdllp_$(param)-scan$(suffix).png"))

    # Also save as PDF for publication quality
    # savefig(plt, joinpath(outdir, "run5-$(scenario)_combdllp_$(param)-scan$(suffix).pdf"))

    return scan_results.best
end

function combination_scan_2d(
    pions_protons::DataFrame,
    range_w::Tuple{Float64,Float64,Float64},
    range_b::Tuple{Float64,Float64,Float64},
    args::Dict{String,Any};
    suffix::String="",
)
    rich_dll_p = pions_protons[:, :RichDLLp]
    torch_dll_p = pions_protons[:, :TorchDLLp]
    labels = pions_protons[:, :label]

    scan_results = parameter_scan(
        rich_dll_p,
        torch_dll_p,
        labels;
        w_range=range_w[1]:range_w[2]:range_w[3],
        b_range=range_b[1]:range_b[2]:range_b[3],
        verbose=false,
    )

    # Visualize AUC
    p1 = visualize_scan_results(scan_results, metric=:auc)

    # Visualize efficiency
    p2 = visualize_scan_results(scan_results, metric=:efficiency)

    # Visualize misidentification probability
    p3 = visualize_scan_results(scan_results, metric=:misid)

    # Visualize purity
    p4 = visualize_scan_results(scan_results, metric=:purity)

    # Combine plots
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))

    outdir = args["output-dir"]
    scenario = args["scenario"]

    if !isempty(suffix)
        suffix = "_" * suffix
    end

    # Save with high quality
    savefig(plt, joinpath(outdir, "run5-$(scenario)_combdllp_scan$(suffix).png"))

    # Also save as PDF for publication quality
    # savefig(plt, joinpath(outdir, "run5-$(scenario)_combdllp_scan$(suffix).pdf"))

    return scan_results.best_auc

end


function performance_curve(
    combined_dll,
    rich_dll_p,
    y,
    args;
    suffix::String="",
)
    pts_combination = misid_eff_dataframe(vec(combined_dll), y; compress=true)
    pts_rich_only = misid_eff_dataframe(rich_dll_p, y; compress=true)

    maskA = pts_combination.misid .> 0.0
    maskB = pts_rich_only.misid .> 0.0

    # Plot overlay
    p = plot(pts_combination.efficiency[maskA], pts_combination.misid[maskA];
        label="RICH + TORCH", lw=2, color=:blue, marker=:none, yaxis=:log10, xticks=0:0.1:1.0)
    plot!(p, pts_rich_only.efficiency[maskB], pts_rich_only.misid[maskB];
        label="RICH", lw=2, color=:red, marker=:none, yaxis=:log10)

    # Axis, scale and style
    xlabel!("Efficiency")
    ylabel!("Mis-id probability")
    xlims!(0.0, 1.0)
    ylims!(1e-2, 1.0)

    # Add specific yticks with formatted decimal labels while keeping log scale
    yticks!([0.01, 0.05, 0.1, 0.5, 1.0], ["0.01", "0.05", "0.1", "0.5", "1.0"])

    title!("p vs π identification")

    outdir = args["output-dir"]
    scenario = args["scenario"]

    if !isempty(suffix)
        suffix = "_" * suffix
    end

    # Save with high quality
    savefig(p, joinpath(outdir, "run5-$(scenario)_combdllp_misid_eff$(suffix).png"))

    # Also save as PDF for publication quality
    # savefig(p, joinpath(outdir, "run5-$(scenario)_combdllp_misid_eff$(suffix).pdf"))
    nothing
end

function main()
    args = parse_args()

    # Assert that args["scenario"] is either "baseline" or "medium"
    if !(args["scenario"] in ["baseline", "medium"])
        error("Invalid scenario: $(args["scenario"]). Valid options are: baseline, medium.")
    end

    println("Loading data from $(args["input-file"])")
    pions_protons = collect_data(args)

    # create output dir if does not exist
    outdir = joinpath(args["output-dir"], args["scenario"])
    if !isdir(outdir)
        mkpath(outdir)
    end
    args["output-dir"] = outdir

    println("Plotting DLLp distributions")
    plot_dll_distributions(pions_protons, args)

    # Perform combination scan
    w_scan = combination_scan_1d(pions_protons, (-100.0, 1.0, 100.0), args, suffix="")
    best_w = w_scan.param

    rich_dll_p = pions_protons[:, :RichDLLp]
    torch_dll_p = pions_protons[:, :TorchDLLp]
    labels = pions_protons[:, :label]

    X, y = prepare_data(
        rich_dll_p,
        torch_dll_p,
        labels
    )

    println("Combined model with w: $(best_w), b: 0")
    model = combination_model(init_w=[1.0 best_w], init_b=[0.0])
    performance_curve(model(X), rich_dll_p, y, args, suffix="opt1")

    #w_fine = combination_scan(pions_protons, (low, step, high), args, suffix="narrow")
    #println("Best w (broad scan): $(best_w)")
    #println("Best w (fine scan): $(w_fine.param)")

    wb_scan = combination_scan_2d(
        pions_protons,
        (-20.0, 1.0, 20.0),
        (-20.0, 1.0, 20.0),
        args,
        suffix="broad"
    )
    best_w = wb_scan.w
    best_b = wb_scan.b

    wb_scan = combination_scan_2d(
        pions_protons,
        (0.0, 0.5, 10.0),
        (-5.0, 0.5, 5.0),
        args,
        suffix="narrow"
    )
    best_w = wb_scan.w
    best_b = wb_scan.b

    println("Combined model with w: $(best_w), b: $(best_b)")
    model = combination_model(init_w=[1.0 best_w], init_b=[best_b])
    performance_curve(model(X), rich_dll_p, y, args, suffix="opt2")

end

main()
