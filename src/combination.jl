# Sigmoid function for clarity and numerical stability
σ(x) = 1 / (1 + exp(-x))


function model(W, b, rich, torch)
    return rich * (W * torch .+ b)
end


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


# Accepts params as a tuple for easy use with Flux
function loss(params, rich, torch, target)
    W, b = params
    pred = model(W, b, rich, torch)
    return auc_surrogate_loss(pred, target)
end
