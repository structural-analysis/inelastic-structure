from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam_static_elastic"
    computational_zero = 1e-10
    isclose_tolerance = 1e-7
    examples_to_test = [
        "simple_beam_static_elastic",
        "simple_beam_static_inelastic",
        "3story_perfect",
        "3story_softening",
        "skew_beams",
        "tripod_corner",
        "tripod_unload",
        "simple_beam_dynamic",
        "simple_beam_dynamic_linear",
        "simple_beam_dynamic_nonlinear_1phase",
        "1story_dynamic",
        "2story_dynamic",
        "torre_dynamic",
    ]


settings = Settings()
