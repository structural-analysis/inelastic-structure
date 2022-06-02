from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "tripod_corner"
    computational_zero = 1e-10


settings = Settings()
