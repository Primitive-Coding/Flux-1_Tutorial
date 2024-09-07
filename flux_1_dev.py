from model import FluxModel


class Flux1Dev(FluxModel):
    def __init__(self) -> None:
        name = "black-forest-labs/FLUX.1-dev"
        save_path = "./SavedModels/Dev"
        model = "Dev"
        super().__init__(name, save_path, model)
