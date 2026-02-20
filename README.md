Launch camera:

```
ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true 
```

Run yolo detection with tf from camera_link to Linear_Inspect (ray cast):

```
ros2 run object_detection yolo_to_pose 
```
