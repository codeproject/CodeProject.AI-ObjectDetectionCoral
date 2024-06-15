import os
import platform

# Import CodeProject.AI SDK
from codeproject_ai_sdk import ModuleOptions

class Settings:
    def __init__(self, model_name: str, model_name_pattern: str, std_model_name: str, 
                 tpu_model_name: str, labels_name: str):
        self.model_name         = model_name
        self.model_name_pattern = model_name_pattern
        self.cpu_model_name     = std_model_name
        self.tpu_model_name     = tpu_model_name
        self.labels_name        = labels_name

        self.MODEL_SEGMENTS = {
                 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq': {
                 #  176.2 ms per inference (5.7 FPS) for 1 TPUs using 1 segment
                 #   97.8 ms per inference (10.2 FPS) for 2 TPUs using 2 segments
                 #   66.2 ms per inference (15.1 FPS) for 3 TPUs using 2 segments
                 #   48.8 ms per inference (20.5 FPS) for 4 TPUs using 1 segment
                 #   37.4 ms per inference (26.8 FPS) for 5 TPUs using 2 segments
                 2: ['all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', 'all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite'],
                 3: ['dumb_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', 'dumb_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite'],
                 5: ['2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite2_448_ptq': {
                 #   60.0 ms per inference (16.7 FPS) for 1 TPUs using 1 segment
                 #   30.6 ms per inference (32.7 FPS) for 2 TPUs using 1 segment
                 #   20.4 ms per inference (49.1 FPS) for 3 TPUs using 2 segments
                 #   17.4 ms per inference (57.4 FPS) for 4 TPUs using 2 segments
                 #   14.5 ms per inference (68.8 FPS) for 5 TPUs using 2 segments
                 3: ['2x_last_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
                 4: ['all_segments_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', 'all_segments_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
                 5: ['166x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '166x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite3_512_ptq': {
                 #   75.7 ms per inference (13.2 FPS) for 1 TPUs using 1 segment
                 #   38.1 ms per inference (26.2 FPS) for 2 TPUs using 1 segment
                 #   26.8 ms per inference (37.3 FPS) for 3 TPUs using 1 segment
                 #   20.7 ms per inference (48.4 FPS) for 4 TPUs using 1 segment
                 #   18.0 ms per inference (55.5 FPS) for 5 TPUs using 1 segment
             },
             'efficientdet_lite3x_640_ptq': {
                 #  181.6 ms per inference (5.5 FPS) for 1 TPUs using 1 segment
                 #   91.5 ms per inference (10.9 FPS) for 2 TPUs using 1 segment
                 #   62.9 ms per inference (15.9 FPS) for 3 TPUs using 2 segments
                 #   49.6 ms per inference (20.2 FPS) for 4 TPUs using 1 segment
                 #   40.4 ms per inference (24.7 FPS) for 5 TPUs using 2 segments
                 3: ['2x_last_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
                 5: ['15x_first_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '15x_first_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5s-int8': {
                 #   36.4 ms per inference (27.5 FPS) for 1 TPUs using 1 segment
                 #   18.9 ms per inference (53.0 FPS) for 2 TPUs using 1 segment
                 #   14.4 ms per inference (69.7 FPS) for 3 TPUs using 1 segment
                 #   11.7 ms per inference (85.4 FPS) for 4 TPUs using 1 segment
                 #   10.8 ms per inference (92.6 FPS) for 5 TPUs using 1 segment
             },
             'yolov5m-int8': {
                 #  100.3 ms per inference (10.0 FPS) for 1 TPUs using 1 segment
                 #   50.5 ms per inference (19.8 FPS) for 2 TPUs using 1 segment
                 #   31.7 ms per inference (31.5 FPS) for 3 TPUs using 2 segments
                 #   26.0 ms per inference (38.5 FPS) for 4 TPUs using 2 segments
                 #   20.1 ms per inference (49.9 FPS) for 5 TPUs using 2 segments
                 3: ['15x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
                 4: ['4x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
                 5: ['4x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5l-int8': {
                 #  182.8 ms per inference (5.5 FPS) for 1 TPUs using 1 segment
                 #   85.6 ms per inference (11.7 FPS) for 2 TPUs using 2 segments
                 #   56.5 ms per inference (17.7 FPS) for 3 TPUs using 2 segments
                 #   43.8 ms per inference (22.8 FPS) for 4 TPUs using 2 segments
                 #   34.0 ms per inference (29.4 FPS) for 5 TPUs using 3 segments
                 2: ['dumb_yolov5l-int8_segment_0_of_2_edgetpu.tflite', 'dumb_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
                 3: ['2x_last_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '2x_last_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
                 4: ['dumb_yolov5l-int8_segment_0_of_2_edgetpu.tflite', 'dumb_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
                 5: ['3x_first_seg_yolov5l-int8_segment_0_of_3_edgetpu.tflite', '3x_first_seg_yolov5l-int8_segment_1_of_3_edgetpu.tflite', '3x_first_seg_yolov5l-int8_segment_2_of_3_edgetpu.tflite'],
             },
             'yolov8s_416_640px': {
                 #   67.5 ms per inference (14.8 FPS) for 1 TPUs using 1 segment
                 #   34.5 ms per inference (29.0 FPS) for 2 TPUs using 1 segment
                 #   22.8 ms per inference (43.8 FPS) for 3 TPUs using 1 segment
                 #   17.0 ms per inference (58.9 FPS) for 4 TPUs using 2 segments
                 #   13.1 ms per inference (76.1 FPS) for 5 TPUs using 2 segments
                 4: ['3x_first_seg_yolov8s_416_640px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov8s_416_640px_segment_1_of_2_edgetpu.tflite'],
                 5: ['3x_first_seg_yolov8s_416_640px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov8s_416_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov8m_416_640px': {
                 #  272.3 ms per inference (3.7 FPS) for 1 TPUs using 1 segment
                 #   95.6 ms per inference (10.5 FPS) for 2 TPUs using 2 segments
                 #   59.5 ms per inference (16.8 FPS) for 3 TPUs using 3 segments
                 #   43.8 ms per inference (22.8 FPS) for 4 TPUs using 2 segments
                 #   35.5 ms per inference (28.2 FPS) for 5 TPUs using 2 segments
                 2: ['all_segments_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
                 3: ['all_segments_yolov8m_416_640px_segment_0_of_3_edgetpu.tflite', 'all_segments_yolov8m_416_640px_segment_1_of_3_edgetpu.tflite', 'all_segments_yolov8m_416_640px_segment_2_of_3_edgetpu.tflite'],
                 4: ['2x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '2x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
                 5: ['3x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov8l_416_640px': {
                 # 1053.4 ms per inference (0.9 FPS) for 1 TPUs using 1 segment
                 #  155.1 ms per inference (6.4 FPS) for 2 TPUs using 2 segments
                 #   98.1 ms per inference (10.2 FPS) for 3 TPUs using 2 segments
                 #   78.3 ms per inference (12.8 FPS) for 4 TPUs using 2 segments
                 #   61.4 ms per inference (16.3 FPS) for 5 TPUs using 2 segments
                 2: ['all_segments_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite'],
                 3: ['15x_first_seg_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite'],
                 4: ['all_segments_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite'],
                 5: ['4x_first_seg_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov8s_640px': {
                 #  541.0 ms per inference (1.8 FPS) for 1 TPUs using 1 segment
                 #   83.7 ms per inference (11.9 FPS) for 2 TPUs using 2 segments
                 #   54.1 ms per inference (18.5 FPS) for 3 TPUs using 3 segments
                 #   40.8 ms per inference (24.5 FPS) for 4 TPUs using 3 segments
                 #   32.9 ms per inference (30.4 FPS) for 5 TPUs using 3 segments
                 2: ['15x_last_seg_yolov8s_640px_segment_0_of_2_edgetpu.tflite', '15x_last_seg_yolov8s_640px_segment_1_of_2_edgetpu.tflite'],
                 3: ['all_segments_yolov8s_640px_segment_0_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_1_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_2_of_3_edgetpu.tflite'],
                 4: ['all_segments_yolov8s_640px_segment_0_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_1_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_2_of_3_edgetpu.tflite'],
                 5: ['all_segments_yolov8s_640px_segment_0_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_1_of_3_edgetpu.tflite', 'all_segments_yolov8s_640px_segment_2_of_3_edgetpu.tflite'],
             },
             'yolov8m_640px': {
                 #  353.8 ms per inference (2.8 FPS) for 1 TPUs using 1 segment
                 #  165.9 ms per inference (6.0 FPS) for 2 TPUs using 2 segments
                 #   95.4 ms per inference (10.5 FPS) for 3 TPUs using 2 segments
                 #   71.9 ms per inference (13.9 FPS) for 4 TPUs using 2 segments
                 #   56.6 ms per inference (17.7 FPS) for 5 TPUs using 2 segments
                 2: ['all_segments_yolov8m_640px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov8m_640px_segment_1_of_2_edgetpu.tflite'],
                 3: ['2x_first_seg_yolov8m_640px_segment_0_of_2_edgetpu.tflite', '2x_first_seg_yolov8m_640px_segment_1_of_2_edgetpu.tflite'],
                 4: ['3x_first_seg_yolov8m_640px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov8m_640px_segment_1_of_2_edgetpu.tflite'],
                 5: ['4x_first_seg_yolov8m_640px_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov8m_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov8l_640px': {
                 # 1517.3 ms per inference (0.7 FPS) for 1 TPUs using 1 segment
                 #  389.8 ms per inference (2.6 FPS) for 2 TPUs using 2 segments
                 #  206.5 ms per inference (4.8 FPS) for 3 TPUs using 2 segments
                 #  149.0 ms per inference (6.7 FPS) for 4 TPUs using 2 segments
                 #  132.4 ms per inference (7.6 FPS) for 5 TPUs using 2 segments
                 2: ['15x_first_seg_yolov8l_640px_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov8l_640px_segment_1_of_2_edgetpu.tflite'],
                 3: ['15x_first_seg_yolov8l_640px_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov8l_640px_segment_1_of_2_edgetpu.tflite'],
                 4: ['2x_last_seg_yolov8l_640px_segment_0_of_2_edgetpu.tflite', '2x_last_seg_yolov8l_640px_segment_1_of_2_edgetpu.tflite'],
                 5: ['2x_first_seg_yolov8l_640px_segment_0_of_2_edgetpu.tflite', '2x_first_seg_yolov8l_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'ipcam-general-v8': {
                 #  241.2 ms per inference (4.1 FPS) for 1 TPUs using 1 segment
                 #   44.7 ms per inference (22.4 FPS) for 2 TPUs using 2 segments
                 #   22.5 ms per inference (44.4 FPS) for 3 TPUs using 2 segments
                 #   16.1 ms per inference (62.0 FPS) for 4 TPUs using 2 segments
                 #   12.2 ms per inference (82.2 FPS) for 5 TPUs using 2 segments
                 2: ['15x_last_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '15x_last_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
                 3: ['166x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '166x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
                 4: ['2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
                 5: ['2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],

             },
        }        

        self.tpu_segments_lists = {}
        if model_name_pattern in self.MODEL_SEGMENTS:
            self.tpu_segments_lists = self.MODEL_SEGMENTS[model_name_pattern]


class Options:

    def __init__(self):

        # ----------------------------------------------------------------------
        # Setup constants

        # Models at:
        # https://coral.ai/models/object-detection/
        # https://github.com/MikeLud/CodeProject.AI-Custom-IPcam-Models/
        # https://github.com/ultralytics/ultralytics
        #
        # YOLOv8 benchmarked with 3 CPU cores and 6 PCIe TPUs
        self.MODEL_SETTINGS = {
            "yolov8": {
                # 59.88 ms throughput / 855.40 ms inference
                "large":  Settings('YOLOv8', 'yolov8l_416_640px',
                                   'yolov8l_416_640px.tflite',                                 # 46Mb CPU
                                   'yolov8l_416_640px_edgetpu.tflite',                         # 48Mb TPU
                                   'coco_labels.txt'),
                # 53.72 ms throughput / 762.86 ms inference
                "medium": Settings('YOLOv8', 'yolov8m_416_640px', \
                                   'yolov8m_416_640px.tflite',                                      # 21Mb CPU
                                   'yolov8m_416_640px_edgetpu.tflite',                              # 22Mb TPU
                                   'coco_labels.txt'),
                # 21.52 ms throughput / 291.35 ms inference
                "small":  Settings('YOLOv8', 'yolov8s_416_640px',
                                   'yolov8s_416_640px.tflite',                                      # 11Mb CPU
                                   'yolov8s_416_640px_edgetpu.tflite',                              # 12Mb TPU
                                   'coco_labels.txt'),
                # 10.35 ms throughput / 123.35 ms inference
                "tiny":   Settings('YOLOv8', 'yolov8n_416_640px',
                                   'yolov8n_416_640px.tflite',                                      # 4Mb CPU
                                   'yolov8n_416_640px_edgetpu.tflite',                              # 3Mb TPU
                                   'coco_labels.txt')
            },
            "yolov5": {
                "large":  Settings('YOLOv5', 'yolov5l-int8',
                                   'yolov5l-int8.tflite',                                      # 46Mb CPU
                                   'yolov5l-int8_edgetpu.tflite',                              # 48Mb TPU
                                   'coco_labels.txt'),
                "medium": Settings('YOLOv5', 'yolov5m-int8',
                                   'yolov5m-int8.tflite',                                      # 21Mb CPU
                                   'yolov5m-int8_edgetpu.tflite',                              # 22Mb TPU
                                   'coco_labels.txt'),
                "small":  Settings('YOLOv5', 'yolov5s-int8',
                                   'yolov5s-int8.tflite',                                      # 7Mb CPU
                                   'yolov5s-int8_edgetpu.tflite',                              # 8Mb TPU
                                   'coco_labels.txt'),
                "tiny":   Settings('YOLOv5', 'yolov5n-int8',
                                   'yolov5n-int8.tflite',                                      # 2Mb CPU
                                   'yolov5n-int8_edgetpu.tflite',                              # 2Mb TPU
                                   'coco_labels.txt')
            },
            "efficientdet-lite": {
                # Large: EfficientDet-Lite3x 90 objects COCO	640x640x3 	2 	197.0 ms 	43.9% mAP
                "large":  Settings('EfficientDet-Lite', 'efficientdet_lite3x_640_ptq', \
                                   'efficientdet_lite3x_640_ptq.tflite',                       # 14Mb CPU
                                   'efficientdet_lite3x_640_ptq_edgetpu.tflite',               # 20Mb TPU
                                   'coco_labels.txt'),
                # Medium: EfficientDet-Lite3 90 objects	512x512x3 	2 	107.6 ms 	39.4% mAP
                "medium": Settings('EfficientDet-Lite', 'efficientdet_lite3_512_ptq', \
                                   'efficientdet_lite3_512_ptq.tflite',                        # CPU
                                   'efficientdet_lite3_512_ptq_edgetpu.tflite',                # TPU
                                   'coco_labels.txt'),
                # Small: EfficientDet-Lite2 90 objects COCO	448x448x3 	2 	104.6 ms 	36.0% mAP
                "small":  Settings('EfficientDet-Lite', 'efficientdet_lite2_448_ptq', \
                                   'efficientdet_lite2_448_ptq.tflite',                        # 10Mb CPU
                                   'efficientdet_lite2_448_ptq_edgetpu.tflite',                # TPU
                                   'coco_labels.txt'),
                # Tiny: EfficientDet-Lite1 90 objects COCO	384x384x3 	2 	56.3 ms 	34.3% mAP
                "tiny":   Settings('EfficientDet-Lite', 'efficientdet_lite1_384_ptq', \
                                   'efficientdet_lite1_384_ptq.tflite',                        # 7Mb CPU
                                   'efficientdet_lite1_384_ptq_edgetpu.tflite',                # TPU
                                   'coco_labels.txt')
            },
            "mobilenet ssd": {
                # Large: SSD/FPN MobileNet V1 90 objects, COCO 640x640x3    TF-lite v2    229.4 ms    31.1% mAP
                "large":  Settings('MobileNet SSD', 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq', \
                                   'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq.tflite',       # CPU
                                   'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite', # TPU
                                   'coco_labels.txt'),
                # Medium: SSDLite MobileDet   90 objects, COCO 320x320x3    TF-lite v1    9.1 ms 	32.9% mAP
                "medium": Settings('MobileNet SSD', 'ssdlite_mobiledet_coco_', \
                                   'ssdlite_mobiledet_coco_qat_postprocess.tflite',            # 5Mb CPU
                                   'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',    # TPU
                                   'coco_labels.txt'),
                # Small: SSD MobileNet V2 90 objects, COCO 300x300x3    TF-lite v2    7.6 ms    22.4% mAP
                "small":  Settings('MobileNet SSD', 'tf2_ssd_mobilenet_v2', \
                                   'tf2_ssd_mobilenet_v2_coco17_ptq.tflite',                   # 6.7Mb CPU
                                   'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite',           # TPU
                                   'coco_labels.txt'),

                # Tiny: MobileNet V2 90 objects, COCO 300x300x3    TF-lite v2 Quant
                "tiny":   Settings('MobileNet SSD', 'ssd_mobilenet_v2_coco_', \
                                   'ssd_mobilenet_v2_coco_quant_postprocess.tflite',           # 6.6Mb CPU
                                   'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',   # TPU
                                   'coco_labels.txt')
            }
        }


        self.ENABLE_MULTI_TPU                   = True
        if platform.system() == 'Darwin':
            self.ENABLE_MULTI_TPU = False

        self.MIN_CONFIDENCE                     = 0.5
        self.INTERPRETER_LIFESPAN_SECONDS       = 3600.0
        self.WATCHDOG_IDLE_SECS                 = 5.0       # To be added to non-multi code
        self.MAX_IDLE_SECS_BEFORE_RECYCLE       = 60.0      # To be added to non-multi code
        self.WARN_TEMPERATURE_THRESHOLD_CELSIUS = 80        # PCIe && Linux only

        self.MAX_PIPELINE_QUEUE_LEN             = 1000      # Multi-only
        self.TILE_OVERLAP                       = 15        # Multi-only.
        self.DOWNSAMPLE_BY                      = 6.0       # Multi-only. Smaller number results in more tiles generated
        self.IOU_THRESHOLD                      = 0.1       # Multi-only

        # ----------------------------------------------------------------------
        # Setup values

        self._show_env_variables = True

        self.module_path    = ModuleOptions.module_path
        self.models_dir     = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{self.module_path}/assets"))
        self.model_name     = os.path.normpath(ModuleOptions.getEnvVariable("CPAI_CORAL_MODEL_NAME", "MobileNet SSD"))
        self.model_size     = ModuleOptions.getEnvVariable("MODEL_SIZE", "Small")   # small, medium, large

        # custom_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("CUSTOM_MODELS_DIR", f"{module_path}/custom-models"))

        self.use_multi_tpu  = ModuleOptions.getEnvVariable("CPAI_CORAL_MULTI_TPU", str(self.ENABLE_MULTI_TPU)).lower() == "true"
        self.min_confidence = float(ModuleOptions.getEnvVariable("MIN_CONFIDENCE", self.MIN_CONFIDENCE))

        self.sleep_time     = 0.01

        # For multi-TPU tiling. Smaller number results in more tiles generated
        self.downsample_by  = float(ModuleOptions.getEnvVariable("CPAI_CORAL_DOWNSAMPLE_BY", self.DOWNSAMPLE_BY))
        self.tile_overlap   = int(ModuleOptions.getEnvVariable("CPAI_CORAL_TILE_OVERLAP",    self.TILE_OVERLAP))
        self.iou_threshold  = float(ModuleOptions.getEnvVariable("CPAI_CORAL_IOU_THRESHOLD", self.IOU_THRESHOLD))

        # Maybe - perhaps! - we need shorter var names
        self.watchdog_idle_secs           = float(ModuleOptions.getEnvVariable("CPAI_CORAL_WATCHDOG_IDLE_SECS",           self.WATCHDOG_IDLE_SECS))
        self.interpreter_lifespan_secs    = float(ModuleOptions.getEnvVariable("CPAI_CORAL_INTERPRETER_LIFESPAN_SECONDS", self.INTERPRETER_LIFESPAN_SECONDS))
        self.max_idle_secs_before_recycle = float(ModuleOptions.getEnvVariable("CPAI_CORAL_MAX_IDLE_SECS_BEFORE_RECYCLE", self.MAX_IDLE_SECS_BEFORE_RECYCLE))
        self.max_pipeline_queue_length    = int(ModuleOptions.getEnvVariable("CPAI_CORAL_MAX_PIPELINE_QUEUE_LEN",         self.MAX_PIPELINE_QUEUE_LEN))
        self.warn_temperature_thresh_C    = int(ModuleOptions.getEnvVariable("CPAI_CORAL_WARN_TEMPERATURE_THRESHOLD_CELSIUS", self.WARN_TEMPERATURE_THRESHOLD_CELSIUS))

        self.set_model(self.model_name)


        # ----------------------------------------------------------------------
        # dump the important variables

        if self._show_env_variables:
            print(f"Debug: MODULE_PATH:           {self.module_path}")
            print(f"Debug: MODELS_DIR:            {self.models_dir}")
            print(f"Debug: CPAI_CORAL_MODEL_NAME: {self.model_name}")
            print(f"Debug: MODEL_SIZE:            {self.model_size}")
            print(f"Debug: CPU_MODEL_NAME:        {self.cpu_model_name}")
            print(f"Debug: TPU_MODEL_NAME:        {self.tpu_model_name}")


    def set_model(self, model_name):
        
        # Normalise input
        self.model_name = model_name.lower()
        if self.model_name not in [ "mobilenet ssd", "efficientdet-lite", "yolov5", "yolov8"]: # 'yolov5' - no sense including v5 anymore
            self.model_name = "mobilenet ssd"

        self.model_size = self.model_size.lower()

        """
        With models MobileNet SSD, EfficientDet-Lite, and YOLOv5/v8, we have 
        three classes of model. The first is basically designed to work in concert
        with the Edge TPU and are compatible with the Dev Board Micro. They are
        very fast and don't require additional CPU resources. The YOLOv5/v8 models
        should be directly comparable with other CPAI modules running YOLOv5/v8.
        They should be high-quality, but are not designed with the Edge TPU in
        mind and rely more heavily on the CPU. The EfficientDet-Lite models are
        in between: not as modern as YOLOv5/v8, but less reliant on the CPU.
        
        Each class of model is broken into four sizes depending on the
        intensity of the workload.
        """
        model_valid = self.model_size in [ "tiny", "small", "medium", "large" ]
        if not model_valid:
            self.model_size = "small"

        # Get settings
		# Note: self.model_name and self.model_size are lowercase to ensure dict lookup works
        settings                = self.MODEL_SETTINGS[self.model_name][self.model_size]
        self.cpu_model_name     = settings.cpu_model_name
        self.tpu_model_name     = settings.tpu_model_name
        self.labels_name        = settings.labels_name

        # pre-chew
        self.model_cpu_file     = os.path.normpath(os.path.join(self.models_dir, self.cpu_model_name))
        self.model_tpu_file     = os.path.normpath(os.path.join(self.models_dir, self.tpu_model_name))
        self.label_file         = os.path.normpath(os.path.join(self.models_dir, self.labels_name))

        self.tpu_segments_lists = {}
        for tpu_cnt, name_list in settings.tpu_segments_lists.items():
            self.tpu_segments_lists[tpu_cnt] = \
                [os.path.normpath(os.path.join(self.models_dir, name)) for name in name_list]
