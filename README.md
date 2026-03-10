# Object Detection

Uses yolo object detection and depth camera to determine position of object.

The center of the bounding box is used with a ray cast to set the tf coordinate.

### Persistent mapping

The tf remains after the object is no longer detected. It uses a `min_distance` and eucledian distance to deduplicate. It also assigns a unique ID to the objects i.e. `linear_board_0`.

```
ros2 run object_detection yolo_to_map 
```

### Non persistent mapping

Simple mapping while object is visible:

```
ros2 run object_detection yolo_to_pose 
```

# Dependencies

pip install ultralytics
pip install openvino openvino-dev