from .baseline import build_compile_baseline
from .model1 import build_compile_model_one


MODEL_BUILDERS = {
    "baseline": build_compile_baseline,
    "model1": build_compile_model_one
}


def get_model(model_name: str, *args, **kwargs):
    """
    Factory to build a model given its name and config arguments.
    
    Args:
        model_name: key in MODEL_BUILDERS
        *args, **kwargs: passed to the builder function
    
    Returns:
        model: compiled model
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    
    builder_fn = MODEL_BUILDERS[model_name]
    return builder_fn(*args, **kwargs)


#model = get_model(cfg.model_name, model_cfg, cfg.training, input_dim=input_shape)



def get_model_builder(model_name: str):
    """
    Return the model builder function (uninstantiated), e.g., for Keras Tuner.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_BUILDERS[model_name]