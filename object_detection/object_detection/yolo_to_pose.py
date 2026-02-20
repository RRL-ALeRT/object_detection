#!/usr/bin/env python

import cv2
import torch
from ultralytics import YOLO
import math
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from tf2_ros import TransformBroadcaster
from image_geometry import PinholeCameraModel
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration
import tf_transformations

from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point

# For AMD ROCm
# os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
# For NVIDIA CUDA
# torch.cuda.set_device(0)

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        
        self.declare_parameter('confidence_threshold', 0.8)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.detections = self.create_publisher(Image, '/yolo_detections', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.camera_model = PinholeCameraModel()
        self.info_sub = self.create_subscription(CameraInfo, '/Spot/kinect_color/camera_info', self.info_callback, 10)

        self.K = None
        self.target_frame = "odom"
        
        self.subscription = self.create_subscription(Image, '/Spot/kinect_color/image_color', self.image_callback, 10)
        self.subscription  # prevent unused variable warning        
        
        package_share_directory = get_package_share_directory('object_detection')
        model_path = os.path.join(package_share_directory, 'models', 'best.pt')
        self.model = YOLO(model_path)  # standard YOLOv8 nano model

        self.latest_depth_image = None
        self.depth_subscription = self.create_subscription(
            Image,
            '/Spot/kinect_range/image',
            self.depth_callback,
            10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape((3, 3)).astype(np.float32)
            self.get_logger().info(f"Camera intrinsics set: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}")

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback(self, frame):
        frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
        results = self.model(frame, stream=True, conf=self.confidence_threshold)
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

                if self.latest_depth_image is not None and self.K is not None:
                    try:
                        height, width = self.latest_depth_image.shape
                        if 0 <= cx < width and 0 <= cy < height:
                            z = float(self.latest_depth_image[cy, cx])
                            if np.issubdtype(self.latest_depth_image.dtype, np.integer):
                                z = z / 1000.0
                            if z == 0.0 or np.isnan(z):
                                self.get_logger().debug("Invalid depth at centroid; skipping")
                                continue

                            fx = float(self.K[0, 0])
                            fy = float(self.K[1, 1])
                            cx_k = float(self.K[0, 2])
                            cy_k = float(self.K[1, 2])

                            X = z
                            Y = -((cx - cx_k) * z / fx)
                            Z = -((cy - cy_k) * z / fy)

                            # Create PointStamped in camera frame
                            point = PointStamped()
                            point.header.stamp = self.get_clock().now().to_msg()
                            point.header.frame_id = "kinect color"  # Use your actual camera frame
                            point.point.x = float(X)
                            point.point.y = float(Y)
                            point.point.z = float(Z)

                            try:
                                # Transform to odom frame
                                transform = self.tf_buffer.lookup_transform(
                                    self.target_frame,
                                    point.header.frame_id,
                                    rclpy.time.Time(),
                                    timeout=Duration(seconds=0.05)
                                )
                                transformed = do_transform_point(point, transform)

                                # Broadcast TF for this detection
                                t = TransformStamped()
                                t.header.stamp = self.get_clock().now().to_msg()
                                t.header.frame_id = self.target_frame
                                t.child_frame_id = "linear_inspection"  # Unique per detection

                                t.transform.translation.x = transformed.point.x
                                t.transform.translation.y = transformed.point.y
                                t.transform.translation.z = transformed.point.z
                                # t.transform.rotation.z = 0.0
                                # t.transform.rotation.w = 1.0

                                self.tf_broadcaster.sendTransform(t)
                                depth_text = f"{z:.2f}m"
                            except Exception as e:
                                self.get_logger().warning(f"TF transform failed: {e}")
                                continue
                    except Exception as e:
                        # self.get_logger().error(f'Error: {e}')
                        pass

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (100, 0, 255)
                thickness = 1
                cv2.putText(frame, f"{r.names[cls]} {confidence} {depth_text}", org, font, fontScale, color, thickness)
                
                # Draw center point
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
        self.detections.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

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