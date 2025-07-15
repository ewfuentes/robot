import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    import scipy.ndimage

    import numpy as np

    from pathlib import Path
    return Path, mo, np, plt, scipy, vd


@app.cell
def _(Path, vd):
    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago')
    _dataset_config = vd.VigorDatasetConfig(panorama_neighbor_radius=1e-6)
    dataset = vd.VigorDataset(_dataset_path, _dataset_config)
    return (dataset,)


@app.cell
def _(np):
    def sample_pts(image_shape):
        gen = np.random.default_rng(0)
        rvs = gen.random((500, 2))
        azimuth_rad = rvs[:, 1] * 2 * np.pi
        elevation_rad = np.arcsin(2 * rvs[:, 0] - 1)

        row_frac = 1.0/2.0 - elevation_rad / np.pi
        row =  row_frac * image_shape[1]
        col =  rvs[:, 1] * image_shape[2]
        return np.stack((row, col), axis=1)
    return (sample_pts,)


@app.cell
def _(dataset, mo, plt, sample_pts):
    _item = dataset[20]
    pts = sample_pts(_item.panorama.shape)
    plt.figure(figsize=(12, 6))
    image = _item.panorama.permute(1, 2, 0).numpy()
    plt.imshow(_item.panorama.permute(1, 2, 0))
    plt.plot(pts[:, 1], pts[:, 0], 'o')
    mo.mpl.interactive(plt.gcf())
    return image, pts


@app.cell
def _():
    # def spherical_pixel_from_azimuth_elevation(azimuth_rad, elev_rad, image_shape):
    #     # 0 degrees azimuth is east
    #     col_frac = (np.pi - azimuth_rad) / ( 2 * np.pi)
    #     row_frac = elev_rad / np.pi + 1 / 2.0

    #     return np.stack([row_frac * image_shape[0], col_frac * image_shape[1]], axis=0)

    # def azimuth_elevation_from_pinhole_pixel(row_frac, col_frac, fov):
    #     # row_frac and col_frac are in [-1, 1]
    #     # compute the distance of the plane
    #     plane_dist_x = 1 / np.tan(fov[0] / 2.0)
    #     plane_dist_y = 1 / np.tan(fov[1] / 2.0)

    #     azimuth_rad = np.arctan2(col_frac, plane_dist_x)
    #     elev_rad = np.arctan2(row_frac, plane_dist_y)
    #     return azimuth_rad, elev_rad
    return


@app.cell
def _(np, scipy):
    CAMERA_FOV_X = np.pi/2.0
    CAMERA_FOV_Y = CAMERA_FOV_X * 1.0

    IMAGE_RES_X = 1024
    IMAGE_RES_Y = int(np.round(IMAGE_RES_X * CAMERA_FOV_Y / CAMERA_FOV_X))

    print(f"{CAMERA_FOV_X=}, {CAMERA_FOV_Y=}, {IMAGE_RES_X=}, {IMAGE_RES_Y}")

    def spherical_pixel_from_azimuth_elevation(azimuth_rad, elev_rad, image_shape):
        # Map azimuth/elevation to pixel coordinates in equirectangular image
        col_frac = (np.pi - azimuth_rad) / (2 * np.pi)
        row_frac = elev_rad / np.pi + 0.5

        row_px = row_frac * image_shape[0]
        col_px = col_frac * image_shape[1]

        return np.stack([row_px, col_px], axis=0)  # shape (2, H_out, W_out)

    def azimuth_elevation_from_pinhole_pixel(row_frac, col_frac, fov):
        # Compute the distance of the plane
        plane_dist_x = 1 / np.tan(fov[0] / 2.0)
        plane_dist_y = 1 / np.tan(fov[1] / 2.0)

        azimuth_rad = np.arctan2(col_frac, plane_dist_x)
        elev_rad = np.arctan2(row_frac, plane_dist_y)
        return azimuth_rad, elev_rad

    def reproject_pinhole(panorama, output_shape, fov, yaw=0.0, pitch=0.0):
        """
        panorama: (H_pano, W_pano, C) — equirectangular panorama
        output_shape: (H_out, W_out)
        fov: (fov_x, fov_y) in radians
        yaw, pitch: camera orientation in radians
        """
        H_out, W_out = output_shape
        fx = 1 / np.tan(fov[0] / 2.0)
        fy = 1 / np.tan(fov[1] / 2.0)

        # Create normalized pixel grid
        row_frac = np.linspace(-1, 1, H_out)
        col_frac = np.linspace(1, -1, W_out)
        row_frac, col_frac = np.meshgrid(row_frac, col_frac, indexing='ij')  # (H_out, W_out)

        # Convert to 3D ray directions (camera coordinates)
        x = col_frac
        y = row_frac
        z = np.ones_like(x)

        dirs = np.stack([
            x,
            y * (fx / fy),  # correct for aspect ratio
            fx * np.ones_like(x)
        ], axis=-1)  # shape (H_out, W_out, 3)

        # Normalize directions
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

        # --- Apply yaw and pitch rotations ---
        # Yaw (around y-axis)
        Ry = np.array([
            [ np.cos(yaw), 0, np.sin(yaw)],
            [          0, 1,          0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        # Pitch (around x-axis)
        Rx = np.array([
            [1,           0,            0],
            [0,  np.cos(pitch), -np.sin(pitch)],
            [0,  np.sin(pitch),  np.cos(pitch)]
        ])
        R = Ry @ Rx  # Combined rotation

        dirs_rot = dirs @ R.T  # shape (H_out, W_out, 3)

        # Convert rotated dirs to azimuth and elevation
        x_rot = dirs_rot[..., 0]
        y_rot = dirs_rot[..., 1]
        z_rot = dirs_rot[..., 2]

        azimuth = np.arctan2(x_rot, z_rot)
        elevation = np.arcsin(y_rot)

        # Map to panorama pixel coordinates
        pano_coords = spherical_pixel_from_azimuth_elevation(azimuth, elevation, panorama.shape[:2])
        pano_coords[1] %= panorama.shape[1]  # wrap
        pano_coords[0] = np.clip(pano_coords[0], 0, panorama.shape[0] - 1)  # clamp

        # Interpolate per channel
        output = np.zeros((H_out, W_out, panorama.shape[2]), dtype=panorama.dtype)
        for c in range(panorama.shape[2]):
            output[..., c] = scipy.ndimage.map_coordinates(
                panorama[..., c],
                pano_coords,
                order=1,
                mode='wrap' if c == 1 else 'nearest'
            )
        return output
    return (
        CAMERA_FOV_X,
        CAMERA_FOV_Y,
        IMAGE_RES_X,
        IMAGE_RES_Y,
        azimuth_elevation_from_pinhole_pixel,
        reproject_pinhole,
        spherical_pixel_from_azimuth_elevation,
    )


@app.cell
def _(
    CAMERA_FOV_X,
    CAMERA_FOV_Y,
    IMAGE_RES_X,
    IMAGE_RES_Y,
    image,
    np,
    reproject_pinhole,
):
    reproj = {y: reproject_pinhole(image, (IMAGE_RES_Y, IMAGE_RES_X), (CAMERA_FOV_X, CAMERA_FOV_Y), yaw=y, pitch=0) for y in np.arange(0, 2*np.pi, np.pi / 4.0)}
    return (reproj,)


@app.cell
def _(mo, plt, reproj):
    plt.figure(figsize=(22,6))

    for i, (yaw, img) in enumerate(reproj.items()):
        plt.subplot(1, len(reproj), i+1)
        plt.imshow(img)
        plt.title(f"{yaw=:0.3f}")
        plt.tight_layout()

    mo.mpl.interactive(plt.gcf())
    return i, img, yaw


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
