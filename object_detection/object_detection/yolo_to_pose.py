#!/usr/bin/env python

import cv2
import torch
from ultralytics import YOLO
import math
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from image_geometry import PinholeCameraModel

from cv_bridge import CvBridge

# For AMD ROCm
# os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
# For NVIDIA CUDA
# torch.cuda.set_device(0)

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        print('Initializing object detection')
        self.bridge = CvBridge()
        
        self.declare_parameter('confidence_threshold', 0.7)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        print('Initializing the yolo annotated image topic')
        self.detections = self.create_publisher(Image, '/yolo_detections', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        print('Initializing input camera topics')
        self.img_cnt = 0
        camera_info = '/Spot/kinect_color/camera_info'
        self.camera_model = PinholeCameraModel()
        self.info_sub = self.create_subscription(CameraInfo, camera_info, self.info_callback, 10)
        
        camera_topic = '/Spot/kinect_color/image_color'
        self.subscription = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.subscription  # prevent unused variable warning        
        
        print('Initializing YOLO')
        model_file = 'best.pt'
        package_share_directory = get_package_share_directory('object_detection')
        model_path = os.path.join(package_share_directory, 'models', model_file)
        self.model = YOLO(model_path)  # standard YOLOv8 nano model

        print('Initializing depth camera topics')
        depth_camera_topic = '/Spot/kinect_range/image'
        self.latest_depth_image = None
        self.depth_subscription = self.create_subscription(
            Image,
            depth_camera_topic,
            self.depth_callback,
            10)
        print('Node initialized with: ')
        print('Model = ' + model_file)
        print('Confidence thres = ' + str(self.confidence_threshold))
        print('Camera topic = ' + camera_topic)
        print('Depth camera topic = ' + depth_camera_topic)
        print('Node successfully initialized')
    def info_callback(self, info_msg):
        self.camera_model.fromCameraInfo(info_msg)

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback(self, frame):
        self.img_cnt = self.img_cnt + 1
        if self.img_cnt < 20:
            return
        print('processing image')
        frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
        results = self.model(frame, stream=True, conf=self.confidence_threshold)
        results_list = list(results)
        if len(results_list) > 0:
            print(str(len(results_list)) + ' object/s detected')
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Put boxes in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 0, 255), 1)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # Optional confidence output in console
                # print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])

                # Optional class name output in console
                # print("Class name -->", r.names[cls])

                # Get depth for center pixel
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                depth_text = "NO DISTANCE"
                
                if self.latest_depth_image is not None:
                    try:
                        height, width = self.latest_depth_image.shape
                        if 0 <= cx < width and 0 <= cy < height:
                            depth_value = self.latest_depth_image[cy, cx]
                            dist_meters = depth_value / 1000.0
                            depth_text = f" {dist_meters:.2f}m"

                            # Broadcast TF
                            if dist_meters > 0 and self.camera_model.tfFrame() is not None:
                                ray = self.camera_model.projectPixelTo3dRay((cx, cy))
                                point_3d = np.array(ray) * dist_meters
                                
                                # Calculate rotation: X axis points to camera (origin)
                                # Vector from Object to Camera is -point_3d
                                x_axis = -point_3d
                                x_axis = x_axis / np.linalg.norm(x_axis)
                                
                                # Up vector (Camera Y is down, so let's use Camera Y as 'up' for calculation to keep orientation stable)
                                # Or use Camera X?
                                # Let's try to keep Z axis horizontal-ish.
                                # If we use up = (0, 1, 0) (Camera Y), then Z = X x Y will be roughly Camera Z.
                                up_axis = np.array([0, 1, 0])
                                
                                # Handle parallel case
                                if np.allclose(x_axis, up_axis) or np.allclose(x_axis, -up_axis):
                                    up_axis = np.array([1, 0, 0])
                                
                                z_axis = np.cross(x_axis, up_axis)
                                z_axis = z_axis / np.linalg.norm(z_axis)
                                
                                y_axis = np.cross(z_axis, x_axis)
                                
                                R = np.column_stack((x_axis, y_axis, z_axis))
                                
                                # Rotation Matrix to Quaternion
                                tr = R[0,0] + R[1,1] + R[2,2]
                                if tr > 0:
                                    S = math.sqrt(tr+1.0) * 2
                                    qw = 0.25 * S
                                    qx = (R[2,1] - R[1,2]) / S
                                    qy = (R[0,2] - R[2,0]) / S
                                    qz = (R[1,0] - R[0,1]) / S
                                elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
                                    S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                                    qw = (R[2,1] - R[1,2]) / S
                                    qx = 0.25 * S
                                    qy = (R[0,1] + R[1,0]) / S
                                    qz = (R[0,2] + R[2,0]) / S
                                elif R[1,1] > R[2,2]:
                                    S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                                    qw = (R[0,2] - R[2,0]) / S
                                    qx = (R[0,1] + R[1,0]) / S
                                    qy = 0.25 * S
                                    qz = (R[1,2] + R[2,1]) / S
                                else:
                                    S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                                    qw = (R[1,0] - R[0,1]) / S
                                    qx = (R[0,2] + R[2,0]) / S
                                    qy = (R[1,2] + R[2,1]) / S
                                    qz = 0.25 * S

                                t = TransformStamped()
                                t.header.stamp = self.get_clock().now().to_msg()
                                t.header.frame_id = self.camera_model.tfFrame() # Replace with map on real robot
                                t.child_frame_id = "Linear_Inspect"
                                t.transform.translation.x = point_3d[0]
                                t.transform.translation.y = point_3d[1]
                                t.transform.translation.z = point_3d[2]
                                t.transform.rotation.x = qx
                                t.transform.rotation.y = qy
                                t.transform.rotation.z = qz
                                t.transform.rotation.w = qw
                                
                                self.tf_broadcaster.sendTransform(t)
                    except Exception as e:
                        # self.get_logger().error(f'Error: {e}')
                        pass

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.8
                color = (100, 0, 255)
                thickness = 1
                cv2.putText(frame, f"{r.names[cls]} {confidence} {depth_text}", org, font, fontScale, color, thickness)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
        self.detections.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        self.img_cnt = 0

def main():
    rclpy.init()
    depth_to_pose_node = DetectionNode()
    try:
        rclpy.spin(depth_to_pose_node)
    except KeyboardInterrupt:
        pass
    depth_to_pose_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
