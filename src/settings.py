from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "2story_dynamic"
    computational_zero = 1e-10
    isclose_tolerance = 1e-7
    examples_to_test = [
        "3story_perfect",
        "3story_softening",
        "simple_beam",
        "skew_beams",
        "tripod_corner",
        "tripod_unload",
    ]


settings = Settings()
