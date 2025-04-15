for epoch in range(num_epochs):
    start = time.perf_counter() # 相比time.time()更加精确
    net.train() # 切换回训练模式
    loss_save = 0
    progress_bar = tqdm(enumerate(train_iter), total=len(train_iter), desc="Train")
    for batch_idx, batch in progress_bar:
        feature, target = batch
        # 清空梯度
        trainer.zero_grad()

        # 切换硬件
        X = feature.to(device)
        Y = target.to(device)

        # 前向传播
        y_pred = net(X)
        _loss = loss(y_pred,Y)

        # 反向传播及更新梯度
        _loss.sum().backward()
        trainer.step()

        # 记录损失值
        loss_now = _loss.sum().item() / feature.shape[0]
        loss_save += loss_now
        progress_bar.set_postfix({"LOSS": loss_now})
    loss_save = loss_save / len(train_iter)
    plt_loss.append(loss_save)
    end = time.perf_counter()
    print(f"epoch{epoch} use time:{end-start}!")
    print(f"loss: {loss_save}")
    use_times += end - start

