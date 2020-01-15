import torch




def save_state_dict(model, model_path, optimizer=None, params:dict=None):
    """
    保存模型的state dict和其他的参数
    ------------------------
    save state dict of model and other params
    """
    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if params is not None:
        checkpoint['params'] = params

    torch.save(checkpoint, model_path)


def load_state_dict2cpu(model, model_path, saved_device='cpu', optimizer=None):
    """
    加载模型的state dict到cpu中，并加载其他的参数
    ----------------------
    load state dict of model to cpu， and load other params
    """
    device = torch.device('cpu')
    if saved_device == 'cpu':
        checkpoint = torch.load(model_path)
    elif saved_device == 'cuda':
        checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('params', None)


def load_state_dict2gpu(model, model_path, saved_device='cuda', optimizer=None):
    """
    加载模型的state dict到gpu中，并加载其他的参数
    ----------------------
    load state dict of model to gpu， and load other params
    """
    if saved_device == 'cuda':
        checkpoint = torch.load(model_path)
    elif saved_device == 'cpu':
        checkpoint = torch.load(model_path, map_location='cuda:0')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('params', None)



def save_multi_models(model_path, **kwargs:dict):
    """
    同时保存多个模型的state dict和其他参数
    ---------------------------------------
    Save state dict and other parameters of multiple models at the same time
    """
    checkpoint = {}
    for key in kwargs.keys():
        checkpoint[key] = {}
        model = kwargs[key]
        length = len(model)

        checkpoint[key]['model_state_dict'] = model[0].state_dict()
        if length >= 2 and model[1] is not None:
            checkpoint[key]['optimizer_state_dict'] = model[1].state_dict()
        if length >= 3 and model[2] is not None:
            checkpoint[key]['params'] = model[2]

    torch.save(checkpoint, model_path)



def load_multi_models2cpu(model_path, saved_device, **kwargs:dict):
    """
    加载多个模型的state dict到cpu中，并加载其他的参数
    ----------------------
    load state dict of multiple models to cpu， and load other params
    """
    device = torch.device('cpu')
    if saved_device == 'cpu':
        checkpoint = torch.load(model_path)
    elif saved_device == 'cuda':
        checkpoint = torch.load(model_path, map_location=device)

    params = {}
    for key in kwargs.keys():
        model = kwargs[key]
        length = len(model)

        model[0].load_state_dict(checkpoint[key]['model_state_dict'])

        if length >=2 and model[1] is not None and 'optimizer_state_dict' in checkpoint:
            model[1].load_state_dict(checkpoint[key]['optimizer_state_dict'])

        params[key] = checkpoint[key].get('params', None)

    return params


def load_multi_models2gpu(model_path, saved_device, **kwargs:dict):
    """
    加载多个模型的state dict到cpu中，并加载其他的参数
    ----------------------
    load state dict of multiple models to cpu， and load other params
    """
    if saved_device == 'cuda':
        checkpoint = torch.load(model_path)
    elif saved_device == 'cpu':
        checkpoint = torch.load(model_path, map_location='cuda:0')

    params = {}
    for key in kwargs.keys():
        model = kwargs[key]
        length = len(model)

        model[0].load_state_dict(checkpoint[key]['model_state_dict'])

        if length >= 2 and model[1] is not None and 'optimizer_state_dict' in checkpoint:
            model[1].load_state_dict(checkpoint[key]['optimizer_state_dict'])

        params[key] = checkpoint[key].get('params', None)

    return params




def save_entire_model(model, model_path):
    """
    保存整个模型
    -------------------
    save entire model
    """
    torch.save(model, model_path)


def load_entire_model(model_path):
    """
    加载整个模型
    -------------------
    load entire model
    """
    model = torch.load(model_path)
    return model


def save_DataParallel_model(model, model_path):
    """
    保存多GPU模型
    ----------------
    Save multi GPU model
    """
    torch.save(model.module.state_dict(), model_path)


