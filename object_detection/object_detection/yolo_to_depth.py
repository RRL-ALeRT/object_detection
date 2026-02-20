#!/usr/bin/env python

import cv2
import torch
from ultralytics import YOLO
import math
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from image_geometry import PinholeCameraModel

from cv_bridge import CvBridge

# For AMD ROCm
# os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
# For NVIDIA CUDA
# torch.cuda.set_device(0)

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.bridge = CvBridge()
        self.detections = self.create_publisher(Image, '/yolo_detections', 10)
        self.distance_pub = self.create_publisher(Float32, '/yolo/distance', 10)
        self.subscription = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning        
        
        package_share_directory = get_package_share_directory('object_detection')
        model_path = os.path.join(package_share_directory, 'models', 'best.pt')
        self.model = YOLO(model_path)  # standard YOLOv8 nano model

        self.latest_depth_image = None
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def image_callback(self, frame):
        frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
        results = self.model(frame, stream=True)
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
                            
                            dist_msg = Float32()
                            dist_msg.data = dist_meters
                            self.distance_pub.publish(dist_msg)
                    except Exception:
                        pass

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (100, 0, 255)
                thickness = 1
                cv2.putText(frame, f"{r.names[cls]} {depth_text}", org, font, fontScale, color, thickness)
                
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