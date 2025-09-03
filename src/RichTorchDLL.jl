module RichTorchDLL

using CairoMakie
using Flux
using ProgressMeter
using Statistics
using StatsBase

export combination_model, constrained_combination_model, calculate_auc
export train_model!, prepare_data
export parameter_scan,
    visualize_scan_results,
    visualize_scan_results!,
    parameter_scan_1d,
    visualize_1d_scan,
    visualize_1d_scan!,
    run_parameter_scan_1d,
    repeated_parameter_scan
export histogram_plot!,
    histogram_plot, multi_histogram!, multi_histogram, histogram_grid, histogram_with_ratio
export trackentry_heatmap
export find_workingpoints,
    misid_eff_points,
    misid_eff_dataframe,
    efficiency_per_momentum_bin,
    efficiency_per_momentum_bin_at_misid_rate,
    fraction_nonzero_per_momentum_bin
export plot_efficiency_vs_momentum,
    efficiency_vs_momentum_for_misid_rate,
    efficiency_vs_momentum_with_per_bin_misid,
    plot_bin_efficiency_data,
    compare_efficiency_vs_momentum,
    compare_bin_efficiency_data,
    compare_per_bin_misid_efficiencies,
    compare_performance_curve

include("combination.jl")
include("training.jl")
include("scan.jl")
include("plotting.jl")
include("trackentry.jl")
include("pidperformance.jl")
include("performance_plots.jl")

end # module RichTorchDLL
