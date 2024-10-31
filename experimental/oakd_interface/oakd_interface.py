
import depthai
import numpy as np
import datetime

def main():
    pipeline = depthai.Pipeline()
    
    # Setup up the IMU
    imu = pipeline.create(depthai.node.IMU)
    imu_out = pipeline.create(depthai.node.XLinkOut)

    imu_out.setStreamName("imu")
    imu.enableIMUSensor(depthai.IMUSensor.ACCELEROMETER_RAW, 250)
    imu.enableIMUSensor(depthai.IMUSensor.GYROSCOPE_RAW, 200)

    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(100)

    imu.out.link(imu_out.input)

    # Setup the main color camera

    cam_rgb = pipeline.create(depthai.node.Camera)

    cam_rgb.setBoardSocket(depthai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(15.0)
    cam_rgb.setSize(1280, 720)
    cam_rgb.initialControl.setFrameSyncMode(depthai.RawCameraControl.FrameSyncMode.INPUT)

    # Setup up the left and right cameras
    cam_left = pipeline.create(depthai.node.Camera)
    cam_left.setBoardSocket(depthai.CameraBoardSocket.CAM_B)
    cam_left.setFps(15.0)
    cam_left.setSize(640, 480)
    cam_left.initialControl.setFrameSyncMode(depthai.RawCameraControl.FrameSyncMode.INPUT)

    cam_right = pipeline.create(depthai.node.Camera)
    cam_right.setBoardSocket(depthai.CameraBoardSocket.CAM_C)
    cam_right.setFps(15.0)
    cam_right.setSize(640, 480)
    cam_right.initialControl.setFrameSyncMode(depthai.RawCameraControl.FrameSyncMode.OUTPUT)


    # Sync the camera images together
    cam_sync = pipeline.create(depthai.node.Sync)
    cam_rgb.video.link(cam_sync.inputs['rgb'])
    cam_left.video.link(cam_sync.inputs['left'])
    cam_right.video.link(cam_sync.inputs['right'])
    cam_sync.setSyncThreshold(datetime.timedelta(milliseconds=10))

    cam_out = pipeline.create(depthai.node.XLinkOut)
    cam_out.setStreamName('cam')
    cam_sync.out.link(cam_out.input)

    with depthai.Device(pipeline) as device:
        imu_queue = device.getOutputQueue(name = 'imu', maxSize=50,  blocking=True)
        cam_queue = device.getOutputQueue(name = 'cam', maxSize=10,  blocking=False)

        while True:
            cam_data = cam_queue.tryGet()
            imu_data = imu_queue.get()
            # print('Received IMU Packet:', imu_data, len(imu_data.packets))
            # print('cam_data', cam_data)

            for packet in imu_data.packets:
                accel = packet.acceleroMeter
                accel_t = accel.getTimestamp()

                gyro = packet.gyroscope
                gyro_t = gyro.getTimestamp()
                # print([accel.sequence, accel.x, accel.y, accel.z], accel_t,
                #       [gyro.sequence, gyro.x, gyro.y, gyro.z], gyro_t)

            if cam_data:
                print(cam_data.getMessageNames(), [(name, data.getSequenceNum())for name, data in cam_data], cam_data.getTimestamp(), cam_data.getIntervalNs())

    ...

if __name__ == "__main__":
    main()
