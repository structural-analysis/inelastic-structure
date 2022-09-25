from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam_dynamic"
    computational_zero = 1e-10


settings = Settings()
