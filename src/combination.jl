function combination_model(; init_w=nothing, init_b=nothing)
    model = Dense(2 => 1)
    if !isnothing(init_w)
        model.weight .= init_w
    end
    if !isnothing(init_b)
        model.bias .= init_b
    end
    return model
end

function constrained_combination_model(; init_w2=0.1f0, init_b=0.0f0)
    model = combination_model(init_w=[1.0f0 - init_w2], init_b=[init_b])

    # Create a wrapper to ensure first weight stays fixed
    function forward(X)
        model.weight[1, 1] = 1.0f0 # reset w1 to 1.0
        return model(X)
    end
    return (model=model, forward=forward)
end

# Sigmoid function definition
σ(x) = 1 / (1 + exp(-x))

# Differentiable AUC surrogate loss (pairwise ranking)
function auc_surrogate_loss(scores, targets)
    pos = findall(targets .== 1)
    neg = findall(targets .== 0)
    loss = 0.0
    for i in pos, j in neg
        loss += σ(scores[j] - scores[i])
    end
    return loss / (length(pos) * length(neg))
end

# Negated loss for maximizing AUC
function loss(model, X, y)
    scores = model(X)
    -auc_surrogate_loss(scores, y)
end
