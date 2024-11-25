
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
import argparse
import numpy as np

from experimental.overhead_matching import spectacular_log_python as slp
import common.time.robot_time_python as rtp

from pathlib import Path

def main(spectacular_log_path: Path, rosbag_out: Path):
    log = slp.SpectacularLog(spectacular_log_path)
    print(log.min_imu_time(), log.max_imu_time())
    print(log.get_imu_sample(log.min_imu_time()).accel_mpss)

    print(log.get_frame(0).rgb_frame().shape)
    print(log.get_frame(0).depth_frame().shape)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Header = typestore.types["std_msgs/msg/Header"]
    Imu = typestore.types["sensor_msgs/msg/Imu"]
    Image = typestore.types["sensor_msgs/msg/Image"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(rosbag_out) as writer:
        topic = '/imu0'
        msgtype = Imu.__msgtype__
        connection = writer.add_connection(topic, msgtype, typestore=typestore)

        t = log.min_imu_time()
        i = 0
        while t < log.max_imu_time():

            sample = log.get_imu_sample(t)
            time_since_epoch = t.time_since_epoch()

            imu_ros = Imu(
                header=Header(
                    seq=i,
                    stamp=Time(
                        sec=int(time_since_epoch.total_seconds()),
                        nanosec = 1000 * time_since_epoch.microseconds),
                    frame_id=''),
                orientation = Quaternion(x=0, y=0, z=0, w=1),
                orientation_covariance = -np.ones((3,3)).flatten(),
                angular_velocity = Vector3(x=sample.gyro_radps[0], y=sample.gyro_radps[1], z=sample.gyro_radps[2]),
                angular_velocity_covariance = np.eye(3).flatten(),
                linear_acceleration = Vector3(x=sample.accel_mpss[0], y=sample.accel_mpss[1], z=sample.accel_mpss[2]),
                linear_acceleration_covariance = np.eye(3).flatten()
            )

            timestamp_ns = int(time_since_epoch.total_seconds()) * 1_000_000 + time_since_epoch.microseconds * 1000
            writer.write(connection, timestamp_ns, typestore.serialize_ros1(imu_ros, msgtype))


            t += rtp.as_duration(0.01)
            i += 1

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to unzipped spectacular_log")
    parser.add_argument("--output", required=True, help="path to output rosbag")

    args = parser.parse_args()

    import ipdb
    with ipdb.launch_ipdb_on_exception():
        main(Path(args.input), Path(args.output))
