from ultralytics import YOLO

# Load your custom model
model = YOLO('dexterity.pt')

# Export to OpenVINO format
# This creates a directory '<model_name>_openvino_model' containing .xml and .bin
model.export(format='openvino')
