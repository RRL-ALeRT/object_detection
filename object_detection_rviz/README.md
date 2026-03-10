# Object Detection RViz Plugin

This package provides an RViz 2 panel for capturing images from a ROS 2 topic.

## Build

```bash
cd ~/ros/robocup
colcon build --packages-select object_detection_rviz
source install/setup.bash
```

## Usage

1. Start RViz 2:
   ```bash
   rviz2
   ```
2. In the top menu, go to **Panels -> Add New Panel**.
3. Select **object_detection_rviz -> CapturePanel**.
4. Valid topic (e.g., `/camera/color/image_raw`) and save directory.
5. Click **Capture Image**.
