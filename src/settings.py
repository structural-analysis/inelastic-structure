from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "3story_softening"
    computational_zero = 1e-10


settings = Settings()
