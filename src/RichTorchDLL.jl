module RichTorchDLL

using Flux
using ProgressMeter

export combination_model,
    constrained_combination_model,
    pairwise_ranking_loss,
    pairwise_ranking_loss_sampled,
    loss,
    calculate_auc,
    calculate_auc_sampled
export train_model!, prepare_data
export parameter_scan, visualize_scan_results
export find_workingpoints, misid_eff_points, misid_eff_dataframe

include("combination.jl")
include("training.jl")
include("scan.jl")
include("pidperformance.jl")

end # module RichTorchDLL
