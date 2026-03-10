from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'object_detection'

# Helper function to recursively find all files in a directory
def generate_data_files(source_dir, target_base):
    data_files = []
    for root, dirs, files in os.walk(source_dir):
        if files:
            # Construct the installation path: share/package_name/target_base/subfolder
            # We replace the local source_dir path with the target installation path
            install_path = os.path.join('share', package_name, root)
            file_list = [os.path.join(root, f) for f in files]
            data_files.append((install_path, file_list))
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + generate_data_files('models', 'models'),
    
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Oliver Tiessen',
    maintainer_email='tiessen@fh-aachen.de',
    description='Object mapping with camera and depth data',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'linear_board = object_detection.linear_board:main',
            'center_depth = object_detection.center_depth:main',
            'yolo_to_depth = object_detection.yolo_to_depth:main',
            'yolo_to_pose = object_detection.yolo_to_pose:main',
            'yolo_to_map = object_detection.yolo_to_map:main',
            'yolo_to_map_openvino = object_detection.yolo_to_map_openvino:main',
            'yolo_low_resource = object_detection.yolo_low_resource:main',
            'capture_image = object_detection.capture_image:main',
        ],
    },
)