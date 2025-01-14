from src.visualization.response import draw_load_disp_history
from src.settings import settings

node_num = 28
dof = 0
each_node_dof_count = 3


if __name__ == "__main__":
    draw_load_disp_history(
        example=settings.example_name,
        node_num=node_num,
        dof=dof,
        each_node_dof_count=each_node_dof_count,
    )
