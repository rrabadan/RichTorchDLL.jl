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

include("combination.jl")
include("training.jl")

end # module RichTorchDLL
