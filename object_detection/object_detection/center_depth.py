import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DepthPixelReader(Node):
    def __init__(self):
        super().__init__('depth_pixel_reader')
        
        # 1. Initialize CvBridge
        self.bridge = CvBridge()

        # 2. Subscribe to the depth topic
        # Adjust the topic name based on your camera launch (e.g., /camera/depth/image_rect_raw)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)
            
        self.get_logger().info('Depth Reader Node has been started.')

    def depth_callback(self, msg):
        try:
            # 3. Convert ROS Image message to OpenCV image
            # 'passthrough' preserves the 16-bit encoding (16UC1)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # 4. Get image dimensions
            height, width = cv_image.shape
            
            # 5. Pick a pixel coordinate (u, v)
            # Here we choose the center of the image
            u = width // 2
            v = height // 2
            
            # 6. Access the depth value
            # Note: Array is accessed as [row, col] -> [v, u]
            depth_value = cv_image[v, u]
            
            # 7. Convert to useful units (RealSense usually outputs millimeters)
            dist_meters = depth_value / 1000.0
            
            self.get_logger().info(f'Center Pixel ({u}, {v}) Depth: {depth_value}mm ({dist_meters:.3f}m)')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthPixelReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()