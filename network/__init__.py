from .models import SegFaceLapa, SegFaceCeleb, SegFaceHelen


def get_model(backbone, input_resolution, model, half=False, backbone_only=False):
    if backbone == "segface_lapa":
        model = SegFaceLapa(input_resolution, model)
    elif backbone == "segface_celeb":
        model = SegFaceCeleb(input_resolution, model, half, backbone_only)
    elif backbone == "segface_helen":
        model = SegFaceHelen(input_resolution, model)
    else:
        raise ValueError("Backbone not implemented")
    return model