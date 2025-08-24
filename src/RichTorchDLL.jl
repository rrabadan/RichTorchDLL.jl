module RichTorchDLL

using Flux
using ProgressMeter

export combination_model, constrained_combination_model, loss, calculate_auc
export train_model!, prepare_data

include("combination.jl")
include("training.jl")

end # module RichTorchDLL
