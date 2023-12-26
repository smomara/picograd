from picograd.nn import MultiLayerPerceptron

xs = [
    [-10.0, 5.0, -15.0],
    [0.0, -20.0, 10.0],
    [15.0, -5.0, 0.0],
    [-5.0, 0.0, 20.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MultiLayerPerceptron(3, [4, 4, 1])


for i in range(100):
    
    # forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    model.zero_grad()
    loss.backward()

    # update
    lr = 0.01
    for p in model.parameters():
        p.data -= lr * p.grad

    print(i+1, loss.data)

print([ypred.data for ypred in ypred])