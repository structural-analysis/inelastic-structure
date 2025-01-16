from enum import Enum
from pydantic import BaseSettings
from tests.examples import Examples


class SiftingType(str, Enum):
    mahini = "mahini"
    not_used = None


class Settings(BaseSettings):
    example_name: str = "plate-ring-innerdisp-soft"
    sifting_type: SiftingType = SiftingType.mahini
    monitor_incremental_disp: bool = True
    controlled_node_for_disp = 28
    controlled_node_dofs_count = 3
    controlled_node_dof_for_disp = 0
    controlled_node_for_mises = 28
    computational_zero = 1e-12
    isclose_tolerance = 1e-7
    output_digits = 10
    examples_to_test = Examples.all
    examples_to_test = list(set(examples_to_test))


settings = Settings()
