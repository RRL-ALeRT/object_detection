from setuptools import find_packages, setup

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/dexterity-model/best.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Oliver Tiessen',
    maintainer_email='tiessen@fh-aachen.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'linear_board = object_detection.linear_board:main',
            'center_depth = object_detection.center_depth:main',
            'yolo_to_depth = object_detection.yolo_to_depth:main',
            'yolo_to_pose = object_detection.yolo_to_pose:main',
        ],
    },
)
