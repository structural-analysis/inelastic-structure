from src.visualization.load_vs_disp import draw_load_disp_history
from src.settings import settings


if __name__ == "__main__":
    draw_load_disp_history(
        example=settings.example_name,
        node_num=settings.controlled_node_for_disp,
        dof=settings.controlled_node_dof_for_disp,
        each_node_dof_count=settings.controlled_node_dofs_count,
    )
