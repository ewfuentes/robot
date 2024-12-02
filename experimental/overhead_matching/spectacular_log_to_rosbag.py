
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
import argparse
import numpy as np

from experimental.overhead_matching import spectacular_log_python as slp
import common.time.robot_time_python as rtp

from pathlib import Path


def main(spectacular_log_path: Path, rosbag_out: Path):
    log = slp.SpectacularLog(spectacular_log_path)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types["std_msgs/msg/Header"]
    Imu = typestore.types["sensor_msgs/msg/Imu"]
    Image = typestore.types["sensor_msgs/msg/Image"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(rosbag_out) as writer:
        imu_conn = writer.add_connection('/imu0', Imu.__msgtype__, typestore=typestore)
        rgb_conn = writer.add_connection(
            '/cam0/image_raw', Image.__msgtype__, typestore=typestore)
        depth_conn = writer.add_connection(
            '/cam0/depth_raw', Image.__msgtype__, typestore=typestore)

        # Write out the IMU Data and Camera Data
        t = log.min_imu_time()
        i = 0

        curr_frame_id = 0
        frame = log.get_frame(curr_frame_id)

        while t < log.max_imu_time():

            sample = log.get_imu_sample(t)
            imu_time_since_epoch = t.time_since_epoch()

            while (frame is not None) and (frame.time_of_validity < t):
                frame_time_since_epoch = frame.time_of_validity.time_since_epoch()
                header = Header(
                    seq=curr_frame_id,
                    stamp=Time(
                        sec=int(frame_time_since_epoch.total_seconds()),
                        nanosec=1000 * frame_time_since_epoch.microseconds),
                    frame_id='cam0'
                )

                rgb_frame = frame.rgb_frame()
                depth_frame = frame.depth_frame().astype(np.uint16)

                rgb_ros = Image(
                    header=header,
                    height=rgb_frame.shape[0],
                    width=rgb_frame.shape[1],
                    encoding='bgr8',
                    is_bigendian=False,
                    step=rgb_frame.shape[1] * 3,
                    data=rgb_frame.flatten()
                )

                depth_ros = Image(
                    header=header,
                    height=depth_frame.shape[0],
                    width=depth_frame.shape[1],
                    encoding='mono16',
                    is_bigendian=False,
                    step=depth_frame.shape[1] * 2,
                    data=depth_frame.flatten().view(np.uint8)
                )

                timestamp_ns = (int(frame_time_since_epoch.total_seconds()) * 1_000_000_000
                                + frame_time_since_epoch.microseconds * 1000)

                writer.write(
                    rgb_conn, timestamp_ns, typestore.serialize_ros1(rgb_ros, Image.__msgtype__))
                writer.write(
                    depth_conn, timestamp_ns, typestore.serialize_ros1(depth_ros, Image.__msgtype__))

                curr_frame_id += 1
                frame = log.get_frame(curr_frame_id)
                
            imu_ros = Imu(
                header=Header(
                    seq=i,
                    stamp=Time(
                        sec=int(imu_time_since_epoch.total_seconds()),
                        nanosec=1000 * imu_time_since_epoch.microseconds),
                    frame_id='imu0'),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
                orientation_covariance=-np.ones((3, 3)).flatten(),
                angular_velocity=Vector3(
                    x=sample.gyro_radps[0], y=sample.gyro_radps[1], z=sample.gyro_radps[2]),
                angular_velocity_covariance=np.eye(3).flatten(),
                linear_acceleration=Vector3(
                    x=sample.accel_mpss[0], y=sample.accel_mpss[1], z=sample.accel_mpss[2]),
                linear_acceleration_covariance=np.eye(3).flatten()
            )

            timestamp_ns = (int(imu_time_since_epoch.total_seconds()) * 1_000_000_000
                            + imu_time_since_epoch.microseconds * 1000)

            writer.write(imu_conn, timestamp_ns, typestore.serialize_ros1(imu_ros, Imu.__msgtype__))

            t += rtp.as_duration(0.01)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to unzipped spectacular_log")
    parser.add_argument("--output", required=True, help="path to output rosbag")

    args = parser.parse_args()

    main(Path(args.input), Path(args.output))
