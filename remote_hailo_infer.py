import os
import cv2
import numpy as np

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("WARNING: hailo_platform not found. Running in MOCK mode.")

class HailoModel:
    def __init__(self, hef_path: str, vdevice=None):
        self.hef_path = hef_path
        self.is_mock = not HAILO_AVAILABLE
        
        if not self.is_mock and os.path.exists(hef_path):
            self.hef = HEF(hef_path)
            # Use shared vdevice if provided, else create one
            self.vdevice = vdevice if vdevice is not None else VDevice()
            self.network_groups = self.vdevice.configure(self.hef, ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe))
            self.network_group = self.network_groups[0]
            self.network_group_params = self.network_group.create_params()
            
            # Create input and output stream parameters
            self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
            # Input info
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.input_shape = self.input_vstream_info.shape # (H, W, C)
        else:
            if not os.path.exists(hef_path):
                print(f"WARNING: HEF file {hef_path} not found. Defaulting to mock.")
                self.is_mock = True
            
            if "yolov8" in hef_path.lower():
                self.input_shape = (640, 640, 3)
            elif "scrfd" in hef_path.lower() or "retina" in hef_path.lower():
                self.input_shape = (640, 640, 3)
            elif "arcface" in hef_path.lower():
                self.input_shape = (112, 112, 3)
            else:
                self.input_shape = (640, 640, 3)

    def infer(self, image: np.ndarray):
        h, w = image.shape[:2]
        input_h, input_w = self.input_shape[0], self.input_shape[1]
        img_resized = cv2.resize(image, (input_w, input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        img_batch = np.expand_dims(img_norm, axis=0)

        if self.is_mock:
            return self._mock_infer(img_batch, self.hef_path)
            
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            input_data = {self.input_vstream_info.name: img_batch}
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_data)
                
            return infer_results

    def _mock_infer(self, img_batch, hef_path):
        if "yolov8" in hef_path.lower():
            return {"yolov8_out": np.array([[[0.5, 0.5, 0.2, 0.6, 0.9, 0]]])} 
        elif "scrfd" in hef_path.lower() or "retina" in hef_path.lower():
            return {"face_out": np.array([[[0.5, 0.5, 0.8, 0.8, 0.95]]])}
        elif "arcface" in hef_path.lower():
            return {"embedding": np.random.rand(1, 512).astype(np.float32)}
        return {}
