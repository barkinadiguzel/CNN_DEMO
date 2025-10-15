def print_summary(model):
    print("Model Summary:")
    for name, param in model.named_parameters():
        print(f"{name:30} {str(param.shape):20} requires_grad={param.requires_grad}")
