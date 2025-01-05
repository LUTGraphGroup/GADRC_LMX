

# 传入的参数依次为模型，优化器，损失函数，数据，训练集边，训练集标签
def model_train(model, optimizer, loss_function, data, train_edge_index, train_label):
    model.train()

    optimizer.zero_grad()   # 梯度清零

    output = model.forward(data)  # 若是C数据集，则output大小为663*409
    link_logits = model.decode(output, train_edge_index)  # 一个一维张量
    # train_label = train_label.to(torch.float32)
    loss = loss_function(link_logits, train_label)
    loss.backward()
    optimizer.step()
    return loss, output






