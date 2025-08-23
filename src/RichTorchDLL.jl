module RichTorchDLL

greet() = print("Hello World!")

using Flux

export model, loss
export train_model

include("combination.jl")
include("training.jl")

end # module RichTorchDLL
