import os
import platform

# Import CodeProject.AI SDK
try:
    from codeproject_ai_sdk import ModuleOptions
except ImportError:
    print("Unable to import CPAI SDK! Faking Options.")
    class ModuleOptions:
        module_path = ''

        def getEnvVariable(env_name, env_def):
            return env_def

class Settings:
    def __init__(self, model_name: str, model_name_pattern: str, std_model_name: str, 
                 tpu_model_name: str, labels_name: str):
        self.model_name         = model_name
        self.model_name_pattern = model_name_pattern
        self.cpu_model_name     = std_model_name
        self.tpu_model_name     = tpu_model_name
        self.labels_name        = labels_name

        self.MODEL_SEGMENTS = {
             'tf2_ssd_mobilenet_v2_coco17_ptq': {
#   6.9 ms/inference (144.3 FPS; 11.7 tensor MPx/sec) for 1 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   3.9 ms/inference (255.1 FPS; 20.8 tensor MPx/sec) for 2 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   2.8 ms/inference (354.6 FPS; 28.8 tensor MPx/sec) for 3 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   2.3 ms/inference (434.8 FPS; 35.4 tensor MPx/sec) for 4 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   2.2 ms/inference (452.5 FPS; 36.7 tensor MPx/sec) for 5 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   2.2 ms/inference (452.5 FPS; 36.7 tensor MPx/sec) for 6 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
             },
             'ssd_mobilenet_v2_coco_quant_postprocess': {
#   7.1 ms/inference (140.1 FPS; 11.4 tensor MPx/sec) for 1 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   3.9 ms/inference (259.1 FPS; 21.0 tensor MPx/sec) for 2 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   2.7 ms/inference (366.3 FPS; 29.8 tensor MPx/sec) for 3 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   2.2 ms/inference (444.4 FPS; 36.1 tensor MPx/sec) for 4 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   2.1 ms/inference (478.5 FPS; 38.9 tensor MPx/sec) for 5 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   2.1 ms/inference (478.5 FPS; 38.9 tensor MPx/sec) for 6 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
             },
             'ssdlite_mobiledet_coco_qat_postprocess': {
#   8.8 ms/inference (113.8 FPS; 10.6 tensor MPx/sec) for 1 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   4.6 ms/inference (217.9 FPS; 20.2 tensor MPx/sec) for 2 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   3.3 ms/inference (305.8 FPS; 28.4 tensor MPx/sec) for 3 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.8 ms/inference (363.6 FPS; 33.9 tensor MPx/sec) for 4 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.8 ms/inference (363.6 FPS; 33.9 tensor MPx/sec) for 5 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.8 ms/inference (363.6 FPS; 33.9 tensor MPx/sec) for 6 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
3: ['15x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', '15x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite'],
4: ['4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', '4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite'],
5: ['4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', '4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite'],
6: ['4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', '4x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite'],
             },
             'ssd_mobilenet_v1_coco_quant_postprocess': {
#   6.7 ms/inference (149.7 FPS; 12.2 tensor MPx/sec) for 1 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   3.5 ms/inference (289.0 FPS; 23.5 tensor MPx/sec) for 2 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.4 ms/inference (411.5 FPS; 33.5 tensor MPx/sec) for 3 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.0 ms/inference (490.2 FPS; 39.8 tensor MPx/sec) for 4 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.0 ms/inference (502.5 FPS; 40.8 tensor MPx/sec) for 5 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.0 ms/inference (505.1 FPS; 41.1 tensor MPx/sec) for 6 TPUs using 2 segments: ssd_mobilenet_v1_coco_quant_postprocess
6: ['dumb_ssd_mobilenet_v1_coco_quant_postprocess_segment_0_of_2_edgetpu.tflite', 'dumb_ssd_mobilenet_v1_coco_quant_postprocess_segment_1_of_2_edgetpu.tflite'],
             },
             'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq': {
# 175.7 ms/inference (  5.7 FPS;  2.2 tensor MPx/sec) for 1 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  89.4 ms/inference ( 11.2 FPS;  4.4 tensor MPx/sec) for 2 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  60.9 ms/inference ( 16.4 FPS;  6.4 tensor MPx/sec) for 3 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  46.3 ms/inference ( 21.6 FPS;  8.4 tensor MPx/sec) for 4 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  38.6 ms/inference ( 25.9 FPS; 10.1 tensor MPx/sec) for 5 TPUs using 2 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  34.5 ms/inference ( 29.0 FPS; 11.3 tensor MPx/sec) for 6 TPUs using 2 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
5: ['2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite'],
6: ['all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', 'all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite0_320_ptq': {
#  23.2 ms/inference ( 43.2 FPS;  4.0 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite0_320_ptq
#  13.9 ms/inference ( 71.7 FPS;  6.7 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   9.8 ms/inference (102.4 FPS;  9.5 tensor MPx/sec) for 3 TPUs using 2 segments: efficientdet_lite0_320_ptq
#   9.2 ms/inference (109.3 FPS; 10.2 tensor MPx/sec) for 4 TPUs using 2 segments: efficientdet_lite0_320_ptq
#   7.7 ms/inference (129.5 FPS; 12.1 tensor MPx/sec) for 5 TPUs using 2 segments: efficientdet_lite0_320_ptq
#   7.5 ms/inference (134.0 FPS; 12.5 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite0_320_ptq
3: ['3x_first_seg_efficientdet_lite0_320_ptq_segment_0_of_2_edgetpu.tflite', '3x_first_seg_efficientdet_lite0_320_ptq_segment_1_of_2_edgetpu.tflite'],
4: ['133x_first_seg_efficientdet_lite0_320_ptq_segment_0_of_2_edgetpu.tflite', '133x_first_seg_efficientdet_lite0_320_ptq_segment_1_of_2_edgetpu.tflite'],
5: ['133x_first_seg_efficientdet_lite0_320_ptq_segment_0_of_2_edgetpu.tflite', '133x_first_seg_efficientdet_lite0_320_ptq_segment_1_of_2_edgetpu.tflite'],
6: ['166x_first_seg_efficientdet_lite0_320_ptq_segment_0_of_2_edgetpu.tflite', '166x_first_seg_efficientdet_lite0_320_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite1_384_ptq': {
#  34.7 ms/inference ( 28.8 FPS;  3.9 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  21.5 ms/inference ( 46.5 FPS;  6.3 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  14.0 ms/inference ( 71.5 FPS;  9.7 tensor MPx/sec) for 3 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  12.3 ms/inference ( 81.5 FPS; 11.1 tensor MPx/sec) for 4 TPUs using 2 segments: efficientdet_lite1_384_ptq
#  11.2 ms/inference ( 89.0 FPS; 12.1 tensor MPx/sec) for 5 TPUs using 2 segments: efficientdet_lite1_384_ptq
#  10.6 ms/inference ( 94.6 FPS; 12.9 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite1_384_ptq
4: ['dumb_efficientdet_lite1_384_ptq_segment_0_of_2_edgetpu.tflite', 'dumb_efficientdet_lite1_384_ptq_segment_1_of_2_edgetpu.tflite'],
5: ['133x_first_seg_efficientdet_lite1_384_ptq_segment_0_of_2_edgetpu.tflite', '133x_first_seg_efficientdet_lite1_384_ptq_segment_1_of_2_edgetpu.tflite'],
6: ['15x_first_seg_efficientdet_lite1_384_ptq_segment_0_of_2_edgetpu.tflite', '15x_first_seg_efficientdet_lite1_384_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite2_448_ptq': {
#  60.6 ms/inference ( 16.5 FPS;  3.1 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite2_448_ptq
#  31.4 ms/inference ( 31.9 FPS;  6.0 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite2_448_ptq
#  20.6 ms/inference ( 48.7 FPS;  9.1 tensor MPx/sec) for 3 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  18.1 ms/inference ( 55.2 FPS; 10.4 tensor MPx/sec) for 4 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  15.3 ms/inference ( 65.4 FPS; 12.3 tensor MPx/sec) for 5 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  14.4 ms/inference ( 69.3 FPS; 13.0 tensor MPx/sec) for 6 TPUs using 3 segments: efficientdet_lite2_448_ptq
3: ['2x_last_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
4: ['4x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '4x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
5: ['166x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '166x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite'],
6: ['15x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_3_edgetpu.tflite', '15x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_3_edgetpu.tflite', '15x_first_seg_efficientdet_lite2_448_ptq_segment_2_of_3_edgetpu.tflite'],
             },
             'efficientdet_lite3_512_ptq': {
#  76.7 ms/inference ( 13.0 FPS;  3.2 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  38.5 ms/inference ( 25.9 FPS;  6.4 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  26.9 ms/inference ( 37.2 FPS;  9.2 tensor MPx/sec) for 3 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  21.1 ms/inference ( 47.3 FPS; 11.7 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  17.5 ms/inference ( 57.3 FPS; 14.2 tensor MPx/sec) for 5 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  17.1 ms/inference ( 58.4 FPS; 14.4 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite3_512_ptq
6: ['2x_last_seg_efficientdet_lite3_512_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_efficientdet_lite3_512_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'efficientdet_lite3x_640_ptq': {
# 180.8 ms/inference (  5.5 FPS;  2.2 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  93.6 ms/inference ( 10.7 FPS;  4.2 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  63.9 ms/inference ( 15.7 FPS;  6.1 tensor MPx/sec) for 3 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  48.1 ms/inference ( 20.8 FPS;  8.1 tensor MPx/sec) for 4 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  40.9 ms/inference ( 24.5 FPS;  9.6 tensor MPx/sec) for 5 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  35.5 ms/inference ( 28.2 FPS; 11.0 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite3x_640_ptq
3: ['2x_last_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '2x_last_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
4: ['all_segments_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', 'all_segments_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
5: ['15x_first_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '15x_first_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
6: ['all_segments_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', 'all_segments_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5n-int8': {
#  27.2 ms/inference ( 36.7 FPS;  6.9 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5n-int8
#  17.2 ms/inference ( 58.2 FPS; 10.9 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5n-int8
#  12.4 ms/inference ( 80.3 FPS; 15.1 tensor MPx/sec) for 3 TPUs using 2 segments: yolov5n-int8
#  12.0 ms/inference ( 83.1 FPS; 15.6 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5n-int8
#  10.9 ms/inference ( 91.4 FPS; 17.1 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5n-int8
#  10.9 ms/inference ( 91.4 FPS; 17.1 tensor MPx/sec) for 6 TPUs using 2 segments: yolov5n-int8
3: ['4x_first_seg_yolov5n-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5n-int8_segment_1_of_2_edgetpu.tflite'],
4: ['133x_first_seg_yolov5n-int8_segment_0_of_2_edgetpu.tflite', '133x_first_seg_yolov5n-int8_segment_1_of_2_edgetpu.tflite'],
5: ['4x_first_seg_yolov5n-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5n-int8_segment_1_of_2_edgetpu.tflite'],
6: ['4x_first_seg_yolov5n-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5n-int8_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5s-int8': {
#  39.9 ms/inference ( 25.1 FPS;  4.7 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5s-int8
#  22.3 ms/inference ( 44.9 FPS;  8.4 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5s-int8
#  15.0 ms/inference ( 66.6 FPS; 12.5 tensor MPx/sec) for 3 TPUs using 2 segments: yolov5s-int8
#  11.7 ms/inference ( 85.5 FPS; 16.0 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5s-int8
#  11.3 ms/inference ( 88.7 FPS; 16.6 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5s-int8
#  10.0 ms/inference (100.4 FPS; 18.8 tensor MPx/sec) for 6 TPUs using 2 segments: yolov5s-int8
3: ['166x_first_seg_yolov5s-int8_segment_0_of_2_edgetpu.tflite', '166x_first_seg_yolov5s-int8_segment_1_of_2_edgetpu.tflite'],
4: ['4x_first_seg_yolov5s-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5s-int8_segment_1_of_2_edgetpu.tflite'],
5: ['166x_first_seg_yolov5s-int8_segment_0_of_2_edgetpu.tflite', '166x_first_seg_yolov5s-int8_segment_1_of_2_edgetpu.tflite'],
6: ['4x_first_seg_yolov5s-int8_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov5s-int8_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5m-int8': {
# 100.9 ms/inference (  9.9 FPS;  1.9 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5m-int8
#  50.6 ms/inference ( 19.8 FPS;  3.7 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5m-int8
#  31.7 ms/inference ( 31.6 FPS;  5.9 tensor MPx/sec) for 3 TPUs using 2 segments: yolov5m-int8
#  25.8 ms/inference ( 38.8 FPS;  7.3 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5m-int8
#  19.9 ms/inference ( 50.1 FPS;  9.4 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5m-int8
#  16.9 ms/inference ( 59.1 FPS; 11.1 tensor MPx/sec) for 6 TPUs using 2 segments: yolov5m-int8
3: ['15x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
4: ['166x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '166x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
5: ['3x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
6: ['15x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '15x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov5l-int8': {
# 183.5 ms/inference (  5.4 FPS;  1.0 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5l-int8
#  85.5 ms/inference ( 11.7 FPS;  2.2 tensor MPx/sec) for 2 TPUs using 2 segments: yolov5l-int8
#  55.2 ms/inference ( 18.1 FPS;  3.4 tensor MPx/sec) for 3 TPUs using 2 segments: yolov5l-int8
#  43.6 ms/inference ( 22.9 FPS;  4.3 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5l-int8
#  34.2 ms/inference ( 29.2 FPS;  5.5 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5l-int8
#  30.0 ms/inference ( 33.3 FPS;  6.2 tensor MPx/sec) for 6 TPUs using 3 segments: yolov5l-int8
2: ['dumb_yolov5l-int8_segment_0_of_2_edgetpu.tflite', 'dumb_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
3: ['2x_last_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '2x_last_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
4: ['3x_first_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
5: ['3x_first_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite'],
6: ['4x_first_seg_yolov5l-int8_segment_0_of_3_edgetpu.tflite', '4x_first_seg_yolov5l-int8_segment_1_of_3_edgetpu.tflite', '4x_first_seg_yolov5l-int8_segment_2_of_3_edgetpu.tflite'],
             },
             'yolov8n_416_640px': {
#  23.7 ms/inference ( 42.1 FPS;  9.7 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8n_384_640px
#  12.1 ms/inference ( 82.9 FPS; 19.1 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8n_384_640px
#   9.1 ms/inference (109.8 FPS; 25.3 tensor MPx/sec) for 3 TPUs using 1 segments: yolov8n_384_640px
#   7.6 ms/inference (131.6 FPS; 30.3 tensor MPx/sec) for 4 TPUs using 1 segments: yolov8n_384_640px
#   7.0 ms/inference (142.2 FPS; 32.8 tensor MPx/sec) for 5 TPUs using 1 segments: yolov8n_384_640px
#   6.5 ms/inference (154.3 FPS; 35.6 tensor MPx/sec) for 6 TPUs using 1 segments: yolov8n_384_640px
             },
             'yolov8s_416_640px': {
#  46.5 ms/inference ( 21.5 FPS;  4.7 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8s_384_608px
#  23.5 ms/inference ( 42.5 FPS;  9.3 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8s_384_608px
#  16.2 ms/inference ( 61.7 FPS; 13.5 tensor MPx/sec) for 3 TPUs using 1 segments: yolov8s_384_608px
#  10.9 ms/inference ( 91.8 FPS; 18.4 tensor MPx/sec) for 4 TPUs using 2 segments: yolov8s_352_608px
#   9.8 ms/inference (102.0 FPS; 22.3 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8s_384_608px
#   8.9 ms/inference (112.0 FPS; 24.5 tensor MPx/sec) for 6 TPUs using 1 segments: yolov8s_384_608px
4: ['4x_first_seg_yolov8s_352_608px_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov8s_352_608px_segment_1_of_2_edgetpu.tflite'],
5: ['3x_first_seg_yolov8s_384_608px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov8s_384_608px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov8m_416_640px': {
# 188.7 ms/inference (  5.3 FPS;  1.1 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8m_352_608px
#  95.1 ms/inference ( 10.5 FPS;  2.6 tensor MPx/sec) for 2 TPUs using 2 segments: yolov8m_416_640px
#  58.7 ms/inference ( 17.0 FPS;  4.3 tensor MPx/sec) for 3 TPUs using 2 segments: yolov8m_416_640px
#  44.0 ms/inference ( 22.7 FPS;  5.7 tensor MPx/sec) for 4 TPUs using 2 segments: yolov8m_416_640px
#  35.5 ms/inference ( 28.1 FPS;  7.0 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8m_416_640px
#  30.9 ms/inference ( 32.4 FPS;  8.1 tensor MPx/sec) for 6 TPUs using 3 segments: yolov8m_416_640px
2: ['all_segments_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
3: ['2x_last_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '2x_last_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
4: ['2x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '2x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
5: ['4x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite'],
6: ['133x_first_seg_yolov8m_416_640px_segment_0_of_3_edgetpu.tflite', '133x_first_seg_yolov8m_416_640px_segment_1_of_3_edgetpu.tflite', '133x_first_seg_yolov8m_416_640px_segment_2_of_3_edgetpu.tflite'],
             },
             'yolov8l_416_640px': {
# 236.2 ms/inference (  4.2 FPS;  0.8 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8l_352_608px
# 118.1 ms/inference (  8.5 FPS;  1.7 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8l_352_608px
#  85.2 ms/inference ( 11.7 FPS;  2.6 tensor MPx/sec) for 3 TPUs using 2 segments: yolov8l_384_608px
#  59.8 ms/inference ( 16.7 FPS;  3.3 tensor MPx/sec) for 4 TPUs using 1 segments: yolov8l_352_608px
#  46.1 ms/inference ( 21.7 FPS;  4.3 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8l_352_608px
#  51.1 ms/inference ( 19.6 FPS;  4.9 tensor MPx/sec) for 6 TPUs using 3 segments: yolov8l_416_640px
3: ['2x_first_seg_yolov8l_384_608px_segment_0_of_2_edgetpu.tflite', '2x_first_seg_yolov8l_384_608px_segment_1_of_2_edgetpu.tflite'],
5: ['4x_first_seg_yolov8l_352_608px_segment_0_of_2_edgetpu.tflite', '4x_first_seg_yolov8l_352_608px_segment_1_of_2_edgetpu.tflite'],
6: ['4x_first_seg_yolov8l_416_640px_segment_0_of_3_edgetpu.tflite', '4x_first_seg_yolov8l_416_640px_segment_1_of_3_edgetpu.tflite', '4x_first_seg_yolov8l_416_640px_segment_2_of_3_edgetpu.tflite'],
             },
             'yolov9t_416_640px': {
#  29.3 ms/inference ( 34.1 FPS;  7.9 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9t_384_640px
#  14.6 ms/inference ( 68.6 FPS; 15.8 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9t_384_640px
#  10.3 ms/inference ( 96.7 FPS; 22.3 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9t_384_640px
#   8.3 ms/inference (120.2 FPS; 27.7 tensor MPx/sec) for 4 TPUs using 1 segments: yolov9t_384_640px
#   7.3 ms/inference (137.7 FPS; 31.8 tensor MPx/sec) for 5 TPUs using 1 segments: yolov9t_384_640px
#   6.6 ms/inference (151.1 FPS; 34.8 tensor MPx/sec) for 6 TPUs using 1 segments: yolov9t_384_640px
             },
             'yolov9s_416_640px': {
#  45.9 ms/inference ( 21.8 FPS;  4.1 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9s_352_576px
#  22.8 ms/inference ( 43.9 FPS;  8.3 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9s_352_576px
#  15.3 ms/inference ( 65.1 FPS; 12.3 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9s_352_576px
#  11.7 ms/inference ( 85.4 FPS; 16.1 tensor MPx/sec) for 4 TPUs using 1 segments: yolov9s_352_576px
#  10.3 ms/inference ( 97.3 FPS; 19.4 tensor MPx/sec) for 5 TPUs using 1 segments: yolov9s_352_608px
#   8.3 ms/inference (120.5 FPS; 22.8 tensor MPx/sec) for 6 TPUs using 2 segments: yolov9s_352_576px
6: ['3x_first_seg_yolov9s_352_576px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov9s_352_576px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov9m_416_640px': {
# 148.0 ms/inference (  6.8 FPS;  1.3 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9m_352_576px
#  73.8 ms/inference ( 13.5 FPS;  2.6 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9m_352_576px
#  49.6 ms/inference ( 20.2 FPS;  3.8 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9m_352_576px
#  37.1 ms/inference ( 26.9 FPS;  5.1 tensor MPx/sec) for 4 TPUs using 1 segments: yolov9m_352_576px
#  35.4 ms/inference ( 28.3 FPS;  6.2 tensor MPx/sec) for 5 TPUs using 1 segments: yolov9m_384_608px
#  33.5 ms/inference ( 29.9 FPS;  7.5 tensor MPx/sec) for 6 TPUs using 2 segments: yolov9m_416_640px
6: ['3x_first_seg_yolov9m_416_640px_segment_0_of_2_edgetpu.tflite', '3x_first_seg_yolov9m_416_640px_segment_1_of_2_edgetpu.tflite'],
             },
             'yolov9c_416_640px': {
# 306.7 ms/inference (  3.3 FPS;  0.8 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9c_416_640px
# 153.2 ms/inference (  6.5 FPS;  1.6 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9c_416_640px
# 103.0 ms/inference (  9.7 FPS;  2.4 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9c_416_640px
#  74.6 ms/inference ( 13.4 FPS;  3.4 tensor MPx/sec) for 4 TPUs using 2 segments: yolov9c_416_640px
#  59.5 ms/inference ( 16.8 FPS;  4.2 tensor MPx/sec) for 5 TPUs using 2 segments: yolov9c_416_640px
#  47.1 ms/inference ( 21.2 FPS;  4.6 tensor MPx/sec) for 6 TPUs using 2 segments: yolov9c_384_608px
4: ['dumb_yolov9c_416_640px_segment_0_of_2_edgetpu.tflite', 'dumb_yolov9c_416_640px_segment_1_of_2_edgetpu.tflite'],
5: ['15x_last_seg_yolov9c_416_640px_segment_0_of_2_edgetpu.tflite', '15x_last_seg_yolov9c_416_640px_segment_1_of_2_edgetpu.tflite'],
6: ['all_segments_yolov9c_384_608px_segment_0_of_2_edgetpu.tflite', 'all_segments_yolov9c_384_608px_segment_1_of_2_edgetpu.tflite'],
             },
             'ipcam-general-v8': {
# 233.2 ms/inference (  4.3 FPS;  1.1 tensor MPx/sec) for 1 TPUs using 1 segments: ipcam-general-v8
#  44.6 ms/inference ( 22.4 FPS;  5.6 tensor MPx/sec) for 2 TPUs using 2 segments: ipcam-general-v8
#  22.7 ms/inference ( 44.1 FPS; 11.1 tensor MPx/sec) for 3 TPUs using 2 segments: ipcam-general-v8
#  16.1 ms/inference ( 62.0 FPS; 15.6 tensor MPx/sec) for 4 TPUs using 2 segments: ipcam-general-v8
#  12.4 ms/inference ( 80.8 FPS; 20.3 tensor MPx/sec) for 5 TPUs using 2 segments: ipcam-general-v8
#  10.5 ms/inference ( 95.5 FPS; 23.9 tensor MPx/sec) for 6 TPUs using 2 segments: ipcam-general-v8
2: ['inc_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', 'inc_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
3: ['all_segments_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', 'all_segments_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
4: ['2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
5: ['3x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '3x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
6: ['2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite'],
             }
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
            "yolov9": {
                "large":  Settings('YOLOv9', 'yolov9c_416_640px',
                                   'yolov9c_416_640px.tflite',                                 # 46Mb CPU
                                   'yolov9c_416_640px_edgetpu.tflite',                         # 48Mb TPU
                                   'coco_labels.txt'),
                "medium": Settings('YOLOv9', 'yolov9m_416_640px', \
                                   'yolov9m_352_576px.tflite',                                      # 21Mb CPU
                                   'yolov9m_352_576px_edgetpu.tflite',                              # 22Mb TPU
                                   'coco_labels.txt'),
                "small":  Settings('YOLOv9', 'yolov9s_416_640px',
                                   'yolov9s_352_576px.tflite',                                      # 11Mb CPU
                                   'yolov9s_352_576px_edgetpu.tflite',                              # 12Mb TPU
                                   'coco_labels.txt'),
                "tiny":   Settings('YOLOv9', 'yolov9t_416_640px',
                                   'yolov9t_384_640px.tflite',                                      # 4Mb CPU
                                   'yolov9t_384_640px_edgetpu.tflite',                              # 3Mb TPU
                                   'coco_labels.txt')
            },
            "yolov8": {
                # 59.88 ms throughput / 855.40 ms inference
                "large":  Settings('YOLOv8', 'yolov8l_416_640px',
                                   'yolov8l_352_608px.tflite',                                 # 46Mb CPU
                                   'yolov8l_352_608px_edgetpu.tflite',                         # 48Mb TPU
                                   'coco_labels.txt'),
                # 53.72 ms throughput / 762.86 ms inference
                "medium": Settings('YOLOv8', 'yolov8m_416_640px', \
                                   'yolov8m_352_608px.tflite',                                      # 21Mb CPU
                                   'yolov8m_352_608px_edgetpu.tflite',                              # 22Mb TPU
                                   'coco_labels.txt'),
                # 21.52 ms throughput / 291.35 ms inference
                "small":  Settings('YOLOv8', 'yolov8s_416_640px',
                                   'yolov8s_384_608px.tflite',                                      # 11Mb CPU
                                   'yolov8s_384_608px_edgetpu.tflite',                              # 12Mb TPU
                                   'coco_labels.txt'),
                # 10.35 ms throughput / 123.35 ms inference
                "tiny":   Settings('YOLOv8', 'yolov8n_416_640px',
                                   'yolov8n_384_640px.tflite',                                      # 4Mb CPU
                                   'yolov8n_384_640px_edgetpu.tflite',                              # 3Mb TPU
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
