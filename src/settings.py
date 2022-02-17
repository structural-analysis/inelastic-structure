from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "skew_beams"


settings = Settings()
