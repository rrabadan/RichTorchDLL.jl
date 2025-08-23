# Negated loss for maximizing AUC
function neg_auc_loss(params, rich, torch, target)
    -loss(params, rich, torch, target)
end

# Training interface
function train_model(params, rich, torch, target; epochs=100, lr=0.01)
    ps = Flux.params(params)
    opt = Descent(lr)
    losses = []
    for epoch = 1:epochs
        gs = gradient(ps) do
            neg_auc_loss(params, rich, torch, target)
        end
        Flux.Optimise.update!(opt, ps, gs)
        l = neg_auc_loss(params, rich, torch, target)
        push!(losses, l)
        println("Epoch $epoch, Loss: $l")
    end
    return params, losses
end
