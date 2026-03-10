import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import datetime
from ament_index_python.packages import get_package_share_directory
import sys

class CaptureAndSaveImage(Node):
    def __init__(self):
        super().__init__('capture_and_save_image')
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        package_src_path = os.path.join(workspace_dir, 'src', 'object_detection', 'image_capture')
        os.makedirs(package_src_path, exist_ok=True)
        self.save_dir = package_src_path

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.save_image_callback,
            10
        )
        self.get_logger().info('Ready to capture and save images.')

    def save_image_callback(self, msg):
        self.get_logger().info('Image received.')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        img_name = f'capture_{timestamp}.jpg'
        img_path = os.path.join(self.save_dir, img_name)
        cv2.imwrite(img_path, cv_image)
        self.get_logger().info(f'Image saved at: {img_path}')

        self.destroy_node()
        sys.exit(0)
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CaptureAndSaveImage()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        sys.exit(0)
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
