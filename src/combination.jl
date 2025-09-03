"""
    combination_model(; init_w=nothing, init_b=nothing)

Creates and returns a combination model, optionally initializing the model's weights (`init_w`) and biases (`init_b`).

# Arguments
- `init_w`: Optional initial weights for the model. Default is `nothing`.
- `init_b`: Optional initial biases for the model. Default is `nothing`.

# Returns
A combination model instance with the specified or default initialization.
"""
function combination_model(; init_w = nothing, init_b = nothing)
    model = Dense(2 => 1)
    if !isnothing(init_w)
        model.weight .= init_w
    end
    if !isnothing(init_b)
        model.bias .= init_b
    end
    return model
end

"""
    constrained_combination_model(; init_w2=0.1f0, init_b=0.0f0)

Creates a constrained combination model with optional initial values for parameters.

# Keyword Arguments
- `init_w2::Float32`: Initial value for the weight parameter `w2`. Defaults to `0.1f0`.
- `init_b::Float32`: Initial value for the bias parameter `b`. Defaults to `0.0f0`.

# Returns
A model object with the specified initial parameters.
"""
function constrained_combination_model(; init_w2 = 0.1f0, init_b = 0.0f0)
    model = combination_model(init_w = [1.0f0, init_w2], init_b = [init_b])

    # Create a wrapper to ensure first weight stays fixed
    function forward(X)
        model.weight[1, 1] = 1.0f0 # reset w1 to 1.0
        return model(X)
    end
    return (model = model, forward = forward)
end


function calculate_auc(scores, labels)

    # Combine and sort data based on scores in descending order
    p = sortperm(scores, rev = true)
    sorted_labels = labels[p]

    n_pos = sum(sorted_labels)
    n_neg = length(labels) - n_pos

    if n_pos == 0 || n_neg == 0
        return 0.5
    end

    # Calculate cumulative sums of true positives and false positives
    tp = cumsum(sorted_labels)
    fp = cumsum(1 .- sorted_labels)

    # Calculate TPR and FPR for all points
    tpr = tp ./ n_pos
    fpr = fp ./ n_neg

    # Add a (0,0) point to the beginning of the curve for proper trapezoidal integration
    tpr = [0; tpr]
    fpr = [0; fpr]

    # Use the trapezoidal rule on the vectors
    # This is equivalent to sum((fpr[2:end] - fpr[1:end-1]) .* (tpr[2:end] + tpr[1:end-1]) / 2)
    auc = sum((fpr[2:end] .- fpr[1:end-1]) .* (tpr[2:end] .+ tpr[1:end-1])) / 2

    return auc
end
