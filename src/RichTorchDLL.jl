module RichTorchDLL

using Flux
using ProgressMeter
using Statistics
using StatsBase

export combination_model,
    constrained_combination_model,
    pairwise_ranking_loss,
    pairwise_ranking_loss_sampled,
    loss,
    calculate_auc,
    calculate_auc_sampled,
    calculate_auc_stratified_sampled
export train_model!, prepare_data
export parameter_scan,
    visualize_scan_results,
    visualize_scan_results!,
    parameter_scan_1d,
    visualize_1d_scan,
    visualize_1d_scan!,
    repeated_parameter_scan
export find_workingpoints, misid_eff_points, misid_eff_dataframe
export histogram_plot!,
    histogram_plot, multi_histogram!, multi_histogram, histogram_grid, histogram_with_ratio
export trackentry_heatmap
export compare_performance_curve,
    efficiency_per_momentum_bin,
    plot_efficiency_vs_momentum,
    efficiency_vs_momentum_for_misid_rate,
    compare_efficiency_vs_momentum

include("combination.jl")
include("training.jl")
include("scan.jl")
include("pidperformance.jl")
include("plotting.jl")
include("trackentry.jl")

end # module RichTorchDLL
