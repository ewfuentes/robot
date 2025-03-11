import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from common.liegroups.se2_python import SE2
    from common.liegroups.so2_python import SO2
    import altair as alt
    import pandas as pd

    import itertools
    return SE2, SO2, alt, itertools, mo, np, pd


@app.cell
def _(SE2, np):
    pts_in_world = np.array([
        [0, -3],
        [3, 0],
        [0, 4],
        [-5, 0],
        # [-3, 3],
    ])
    robot_from_world = SE2(np.array([2, 0]))

    return pts_in_world, robot_from_world


@app.cell
def _(SO2, itertools, np, pd, pts_in_world, robot_from_world):
    def compute_zero_error_contour(a_idxs, pts_in_world, robot_from_world):
        # Let the length a be between two landmarks whose indices are chosen with a_idxs
        # Let the angle alpha be the angle between the two observation bearings
        # The angle beta be the angle at the first landmark between the second landmark and the robot.
        # An angle of zero corresponds to the second landmark and increases counter clockwise
        pts_in_robot = np.stack([robot_from_world * x for x in pts_in_world])
        bearing_in_robot = np.arctan2(pts_in_robot[:, 1], pts_in_robot[:, 0])
        length_a = np.linalg.norm(pts_in_world[a_idxs[0]] - pts_in_world[a_idxs[1]])
        alpha_rad = np.minimum(np.abs(bearing_in_robot[a_idxs[0]] - bearing_in_robot[a_idxs[1]]), 
                       np.abs(bearing_in_robot[a_idxs[1]] - bearing_in_robot[a_idxs[0]]))
        sin_alpha = np.sin(alpha_rad)
        ratio = sin_alpha / length_a
    
        beta_rad = np.linspace(0, np.pi - alpha_rad, 100)
        gamma_rad = np.pi - alpha_rad - beta_rad
    
        length_b = np.sin(beta_rad) / ratio
        length_c = np.sin(gamma_rad) / ratio
    
        b_in_a = pts_in_world[a_idxs[1]] - pts_in_world[a_idxs[0]]
        b_in_a_dir = b_in_a / length_a
    
        robots_in_world = []

        for i in range(beta_rad.shape[0]):
            robots_in_world.append(SO2(beta_rad[i]) * b_in_a_dir * length_c[i] + pts_in_world[a_idxs[0]])
    
        return np.stack(robots_in_world)

    dfs = []
    for a_idxs in itertools.combinations(range(pts_in_world.shape[0]), r=2):
        robot_in_world = compute_zero_error_contour(a_idxs, pts_in_world, robot_from_world)
        dfs.append(pd.DataFrame({
            'x': robot_in_world[:, 0],
            'y': robot_in_world[:, 1],
            'pair': [a_idxs] * robot_in_world.shape[0],
            'index': list(range(robot_in_world.shape[0]))}))
    robots_in_world = pd.concat(dfs)
    
    

    return (
        a_idxs,
        compute_zero_error_contour,
        dfs,
        robot_in_world,
        robots_in_world,
    )


@app.cell
def _(a_idxs, alt, mo, robots_in_world):
    a_idxs
    
    chart = mo.ui.altair_chart(
        alt.Chart(robots_in_world)
            .mark_line()
            .encode(
                x=alt.X('x', sort=None, scale=alt.Scale(domain=[-4, 4])),
                y=alt.Y('y', scale=alt.Scale(domain=[-4, 4])),
                color='pair',
                order='index')
            .properties(
                width=600,
                height=600
            ))

    mo.vstack([chart])

    return (chart,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
