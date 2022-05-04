from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "skew_beams"
    abar0: float = 0.15


settings = Settings()
