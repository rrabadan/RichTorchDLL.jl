function combination_model()
    return Dense(2 => 1)
end

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
