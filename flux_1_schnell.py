from model import FluxModel


class Flux1Schnell(FluxModel):
    def __init__(self) -> None:
        name = "black-forest-labs/FLUX.1-schnell"
        save_path = "./SavedModels/Schnell"
        model = "Schnell"
        super().__init__(name, save_path, model)
