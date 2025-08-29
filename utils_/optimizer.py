from torch import optim

def get_optimizer(config, model):
    params = [p for p in model.parameters()]
    if config.optimizer == 'Adam':
        return optim.Adam(params, lr=config.lr, weight_decay = config.weight_decay)
    elif config.optimizer == 'SGD':
        return optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)

    