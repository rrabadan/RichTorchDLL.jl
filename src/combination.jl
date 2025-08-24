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

function constrained_combination_model(; init_w2 = 0.1f0, init_b = 0.0f0)
    model = combination_model(init_w = [1.0f0 - init_w2], init_b = [init_b])

    # Create a wrapper to ensure first weight stays fixed
    function forward(X)
        model.weight[1, 1] = 1.0f0 # reset w1 to 1.0
        return model(X)
    end
    return (model = model, forward = forward)
end

# Sigmoid function definition
σ(x) = 1 / (1 + exp(-x))

# Differentiable AUC surrogate loss (pairwise ranking)
function pairwise_ranking_loss(scores, targets)
    pos_indices = findall(targets .== 1)
    neg_indices = findall(targets .== 0)

    if isempty(pos_indices) || isempty(neg_indices)
        return 0.0  # Handle edge case
    end

    # Extract scores
    pos_scores = scores[pos_indices]
    neg_scores = scores[neg_indices]

    # Compute all differences using broadcasting
    diffs = neg_scores .- permutedims(pos_scores)

    # Apply sigmoid to all differences at once
    sig_diffs = σ.(diffs)

    # Sum and normalize
    return sum(sig_diffs) / (length(pos_indices) * length(neg_indices))
end

function loss(model, X, y)
    scores = vec(model(X))
    return pairwise_ranking_loss(scores, y)
end

# AUC calculation (for evaluation)
function calculate_auc(scores, targets)
    pos = findall(targets .== 1)
    neg = findall(targets .== 0)

    # Count correct pairwise rankings
    correct = 0
    total = length(pos) * length(neg)

    for i in pos, j in neg
        if scores[i] > scores[j]
            correct += 1
        elseif scores[i] == scores[j]
            correct += 0.5  # Tie counts as half correct
        end
    end

    return correct / total
end
