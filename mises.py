from src.visualization.mises_vs_disp import draw_mises_vs_disp_history
from src.settings import settings


if __name__ == "__main__":
    draw_mises_vs_disp_history(
        example=settings.example_name,
        disp_node_num=settings.controlled_node_for_disp,
        mises_node_num=settings.controlled_node_for_mises,
        disp_dof=settings.controlled_node_dof_for_disp,
    )
