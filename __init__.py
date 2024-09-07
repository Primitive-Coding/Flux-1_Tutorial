from flux_1_schnell import Flux1Schnell
from flux_1_dev import Flux1Dev


if __name__ == "__main__":

    schnell = Flux1Schnell()
    # dev = Flux1Dev()
    # dev.save_model()
    schnell.get_image(
        "Create a hyper-realistic from the view of the Moon, an asteroid striking Earth",
        "asteroid",
    )
