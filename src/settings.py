from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "tripod_unload"
    computational_zero = 1e-10


settings = Settings()
