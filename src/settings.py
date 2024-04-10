from enum import Enum
from pydantic import BaseSettings

from tests.examples import Examples


class SiftingType(str, Enum):
    mahini = "mahini"
    not_used = None


class Settings(BaseSettings):
    example_name: str = "plate-16element-elastic-q8r-distributed-navier-verify"
    sifting_type: SiftingType = SiftingType.mahini
    computational_zero = 1e-12
    isclose_tolerance = 1e-7
    output_digits = 10
    examples_to_test = list(set(Examples.plates))


settings = Settings()
