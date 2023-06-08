from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam"
    computational_zero = 1e-10
    isclose_tolerance = 1e-7
    examples_to_test = [
        "3story_perfect",
        "3story_softening",
        "simple_beam",
        "skew_beams",
        "tripod_corner",
        "tripod_unload",
        "simple_beam_dynamic",
        "simple_beam_dynamic_linear",
        "1story_dynamic",
        "2story_dynamic",
        "torre_dynamic",
    ]


settings = Settings()
