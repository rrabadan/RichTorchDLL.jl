using Flux: train!, Descent

function train_model!(model, X, y; epochs=100, lr=0.01)
    opt = Descent(lr)
    losses = []
    for epoch = 1:epochs
        train!(loss, model, [(X, y)], opt)
        current_loss = loss(model, X, y)
        push!(losses, current_loss)
        println("Epoch $epoch, Loss: $current_loss")
    end
    return losses
end
