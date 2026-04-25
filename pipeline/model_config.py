"""Model-family configuration for generation/extraction compatibility."""


MODEL_THINKING_BOUNDARIES = {
    "qwen": "</think>",
}


def local_model_id(model_id: str | None) -> str | None:
    """Drop an HF provider suffix such as ':nscale' from a model id."""
    if model_id is None:
        return None
    return model_id.split(":", 1)[0]


def thinking_boundary_for_model(model_id: str | None) -> str | None:
    """Return the serialized thinking/final-response boundary for a model."""
    local_id = local_model_id(model_id)
    if local_id is None:
        return None
    lowered = local_id.lower()
    for family, boundary in MODEL_THINKING_BOUNDARIES.items():
        if family in lowered:
            return boundary
    return None
