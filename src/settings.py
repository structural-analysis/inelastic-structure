from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "1story_dynamic_inelastic_ll1.0_ap400k"
    computational_zero = 1e-10
    isclose_tolerance = 1e-7
    examples_to_test = [
        "simple_beam_static_elastic",
        "simple_beam_static_inelastic",
        "torre_static_elastic",
        "skew_beams_static_elastic",
        "tripod_corner",
        "tripod_unload",
        "3story_static_elastic",
        "3story_static_inelastic_perfect",
        "3story_static_inelastic_softening",
        "torre_dynamic_elastic",
        "simple_beam_dynamic_elastic",
        "1story_dynamic_elastic",
        "2story_dynamic_elastic",
        "simple_beam_dynamic_inelastic_1phase",
    ]


settings = Settings()
