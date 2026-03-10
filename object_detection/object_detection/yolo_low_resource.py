#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration
from tf2_geometry_msgs import do_transform_point
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

# Import your custom messages
from world_info_msgs.msg import BoundingBox, BoundingBoxArray

class FastDetectionNode(Node):
    def __init__(self):
        super().__init__('fast_detection_node')
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('confidence_threshold', 0.8)
        self.declare_parameter('distance_threshold', 10.0)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value

        # Publishers (Using custom message instead of Image)
        self.bb_pub = self.create_publisher(BoundingBoxArray, '/yolo_detections/bb', 1)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera Info & Intrinsics
        self.K = None
        self.target_frame = "odom"
        self.create_subscription(CameraInfo, '/Spot/kinect_color/camera_info', self.info_callback, 1)

        # Subscriptions (Queue size 1 to drop frames if lagging)
        self.create_subscription(Image, '/Spot/kinect_color/image_color', self.image_callback, 1)
        self.latest_depth_image = None
        self.create_subscription(Image, '/Spot/kinect_range/image', self.depth_callback, 1)

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Tracking
        self.tracked_objects = {}
        self.next_object_id = 0

        # Load Model
        package_share_directory = get_package_share_directory('object_detection')
        model_path = os.path.join(package_share_directory, 'models', 'dexterity_openvino_model')
        self.model = YOLO(model_path, task='detect')
        self.get_logger().info(f"Model loaded from {model_path}")

    def info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape((3, 3)).astype(np.float32)

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def euclidean_distance(self, pos1, pos2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

    def update_or_create_object(self, position, class_name, class_id):
        closest_id = None
        min_distance = self.distance_threshold
        
        for obj_id, obj_data in self.tracked_objects.items():
            distance = self.euclidean_distance(position, obj_data['position'])
            if distance < min_distance:
                min_distance = distance
                closest_id = obj_id
        
        if closest_id is not None:
            self.tracked_objects[closest_id]['position'] = position
            return self.tracked_objects[closest_id]['frame_id'], closest_id
        else:
            new_id = self.next_object_id
            self.next_object_id += 1
            frame_id = f"{class_name}_{new_id}"
            self.tracked_objects[new_id] = {'position': position, 'frame_id': frame_id, 'class': class_id}
            return frame_id, new_id

    def image_callback(self, frame_msg):
        # 1. OPTIMIZATION: Look up camera->odom transform ONCE per frame, not per object
        camera_frame_id = frame_msg.header.frame_id
        try:
            cam_to_odom_tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.02)
            )
        except Exception as e:
            self.get_logger().warning(f"TF lookup skipped this frame: {e}")
            return # Skip processing if we don't know where the camera is

        # 2. Run Inference
        cv_image = self.bridge.imgmsg_to_cv2(frame_msg, "bgr8")
        results = self.model.predict(cv_image, verbose=False, conf=self.confidence_threshold)

        # Initialize BoundingBoxArray message
        bb_array_msg = BoundingBoxArray()
        bb_array_msg.header = frame_msg.header
        # bb_array_msg.type = "your_model_type" # Uncomment if your msg definition uses this

        if len(results) == 0 or len(results[0].boxes) == 0:
            self.bb_pub.publish(bb_array_msg)
            return

        result = results[0]
        
        # 3. Process Detections
        for box in result.boxes:
            # Extract 2D info
            cx, cy, width, height = box.xywh.cpu()[0]
            cx, cy = int(cx.item()), int(cy.item())
            confidence = float(box.conf.cpu()[0].item())
            cls_id = int(box.cls.cpu()[0].item())
            cls_name = result.names[cls_id]

            # Populate custom 2D bounding box message
            bb_msg = BoundingBox()
            bb_msg.name = cls_name
            bb_msg.confidence = confidence
            bb_msg.width = float(width)
            bb_msg.height = float(height)
            bb_msg.cx = float(cx)
            bb_msg.cy = float(cy)
            bb_array_msg.array.append(bb_msg)

            # 4. Depth & 3D Math (Only if depth is available)
            if self.latest_depth_image is not None and self.K is not None:
                h, w = self.latest_depth_image.shape
                if 0 <= cx < w and 0 <= cy < h:
                    z = float(self.latest_depth_image[cy, cx])
                    if np.issubdtype(self.latest_depth_image.dtype, np.integer):
                        z /= 1000.0  # Convert mm to meters
                    
                    if z > 0.0 and not np.isnan(z):
                        # Pinhole projection math
                        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
                        cx_k, cy_k = float(self.K[0, 2]), float(self.K[1, 2])
                        
                        X = z
                        Y = -((cx - cx_k) * z / fx)
                        Z = -((cy - cy_k) * z / fy)

                        # Create point and apply the pre-fetched transform
                        point = PointStamped()
                        point.point.x, point.point.y, point.point.z = X, Y, Z
                        transformed = do_transform_point(point, cam_to_odom_tf)
                        position = (transformed.point.x, transformed.point.y, transformed.point.z)

                        # Track and update object
                        frame_id, obj_id = self.update_or_create_object(position, cls_name, cls_id)

        # 5. Broadcast TFs for all tracked objects
        for obj_id, obj_data in self.tracked_objects.items():
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.target_frame
            t.child_frame_id = obj_data['frame_id']
            t.transform.translation.x = obj_data['position'][0]
            t.transform.translation.y = obj_data['position'][1]
            t.transform.translation.z = obj_data['position'][2]
            self.tf_broadcaster.sendTransform(t)

        # 6. Publish the lightweight custom message!
        self.bb_pub.publish(bb_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FastDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()