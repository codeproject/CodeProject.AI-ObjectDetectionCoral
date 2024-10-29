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

        # Tested on a HP EliteDesk G4 800 SFF i5-8500 3.0 GHz
        self.MODEL_SEGMENTS = {
             'tf2_ssd_mobilenet_v2_coco17_ptq': {
#   6.6 ms/inference (151.1 FPS; 12.3 tensor MPx/sec) for 1 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   3.4 ms/inference (295.0 FPS; 24.0 tensor MPx/sec) for 2 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   2.4 ms/inference (416.7 FPS; 33.9 tensor MPx/sec) for 3 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   1.9 ms/inference (540.5 FPS; 43.8 tensor MPx/sec) for 4 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   1.8 ms/inference (565.0 FPS; 46.0 tensor MPx/sec) for 5 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   1.6 ms/inference (609.8 FPS; 49.6 tensor MPx/sec) for 6 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   1.6 ms/inference (645.2 FPS; 52.6 tensor MPx/sec) for 7 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
#   1.5 ms/inference (666.7 FPS; 54.1 tensor MPx/sec) for 8 TPUs using 1 segments: tf2_ssd_mobilenet_v2_coco17_ptq
             '_tflite': ('all_segments_tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite', '2e4d39bd76ccbf6fa3b7400a2fb0b8e0')
             },
             'ssd_mobilenet_v2_coco_quant_postprocess': {
#   6.2 ms/inference (161.6 FPS; 13.1 tensor MPx/sec) for 1 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   3.1 ms/inference (317.5 FPS; 25.8 tensor MPx/sec) for 2 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   2.2 ms/inference (454.5 FPS; 36.9 tensor MPx/sec) for 3 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   1.8 ms/inference (552.5 FPS; 44.8 tensor MPx/sec) for 4 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   1.7 ms/inference (591.7 FPS; 48.1 tensor MPx/sec) for 5 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   1.4 ms/inference (740.7 FPS; 60.3 tensor MPx/sec) for 6 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   1.3 ms/inference (775.2 FPS; 63.0 tensor MPx/sec) for 7 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
#   1.3 ms/inference (775.2 FPS; 63.0 tensor MPx/sec) for 8 TPUs using 1 segments: ssd_mobilenet_v2_coco_quant_postprocess
             '_tflite': ('all_segments_ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', '02baf9c3bb521f6555cdecabea32cbb0')
             },
             'ssdlite_mobiledet_coco_qat_postprocess': {
#   7.4 ms/inference (135.5 FPS; 12.6 tensor MPx/sec) for 1 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   3.8 ms/inference (266.7 FPS; 24.8 tensor MPx/sec) for 2 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.7 ms/inference (375.9 FPS; 35.0 tensor MPx/sec) for 3 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.3 ms/inference (434.8 FPS; 40.5 tensor MPx/sec) for 4 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.2 ms/inference (458.7 FPS; 42.7 tensor MPx/sec) for 5 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.1 ms/inference (483.1 FPS; 44.9 tensor MPx/sec) for 6 TPUs using 2 segments: ssdlite_mobiledet_coco_qat_postprocess
#   2.0 ms/inference (490.2 FPS; 45.6 tensor MPx/sec) for 7 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
#   1.9 ms/inference (518.1 FPS; 48.3 tensor MPx/sec) for 8 TPUs using 1 segments: ssdlite_mobiledet_coco_qat_postprocess
5: [('166x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', 'de0060c0d5bd8e4d24fc7ea6515335e6'), ('166x_first_seg_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite', '2c6e6a85e9e2a4d5db8d690904a4488d')],
6: [('dumb_ssdlite_mobiledet_coco_qat_postprocess_segment_0_of_2_edgetpu.tflite', 'cd34598bdfae8f0d1af0b4e2161941b8'), ('dumb_ssdlite_mobiledet_coco_qat_postprocess_segment_1_of_2_edgetpu.tflite', '38af20020bc363eeb8f68e70cd951e46')],
             '_tflite': ('all_segments_ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite', '6d3fa7e552b9c6f58b31237772d83389')
             },
             'ssd_mobilenet_v1_coco_quant_postprocess': {
#   5.6 ms/inference (178.6 FPS; 14.5 tensor MPx/sec) for 1 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.8 ms/inference (352.1 FPS; 28.6 tensor MPx/sec) for 2 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   2.0 ms/inference (490.2 FPS; 39.9 tensor MPx/sec) for 3 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   1.7 ms/inference (578.0 FPS; 47.1 tensor MPx/sec) for 4 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   1.6 ms/inference (621.1 FPS; 50.3 tensor MPx/sec) for 5 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   1.4 ms/inference (694.4 FPS; 56.3 tensor MPx/sec) for 6 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   1.4 ms/inference (714.3 FPS; 58.0 tensor MPx/sec) for 7 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
#   1.4 ms/inference (714.3 FPS; 58.0 tensor MPx/sec) for 8 TPUs using 1 segments: ssd_mobilenet_v1_coco_quant_postprocess
             '_tflite': ('all_segments_ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite', '9cf9b99a2ebaf703ca598f2d5a9b1cdf')
             },
             'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq': {
# 175.0 ms/inference (  5.7 FPS;  2.2 tensor MPx/sec) for 1 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  88.3 ms/inference ( 11.3 FPS;  4.4 tensor MPx/sec) for 2 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  59.7 ms/inference ( 16.8 FPS;  6.5 tensor MPx/sec) for 3 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  45.2 ms/inference ( 22.1 FPS;  8.6 tensor MPx/sec) for 4 TPUs using 1 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  35.8 ms/inference ( 27.9 FPS; 10.9 tensor MPx/sec) for 5 TPUs using 2 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  33.1 ms/inference ( 30.2 FPS; 11.8 tensor MPx/sec) for 6 TPUs using 2 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  29.5 ms/inference ( 33.9 FPS; 13.2 tensor MPx/sec) for 7 TPUs using 2 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
#  25.4 ms/inference ( 39.4 FPS; 15.4 tensor MPx/sec) for 8 TPUs using 3 segments: tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq
5: [('2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', '6d131c01fd57097c484dfa3c9c98bd2f'), ('2x_last_seg_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite', '9034af141c83e438fc8ebc5a22d5aa94')],
6: [('all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', 'e0fd603108e96ecf24cae51e28895e60'), ('all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite', 'ca859ab42ab6fc81efee5665b32db394')],
7: [('all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_2_edgetpu.tflite', 'e0fd603108e96ecf24cae51e28895e60'), ('all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_2_edgetpu.tflite', 'ca859ab42ab6fc81efee5665b32db394')],
8: [('dumb_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_0_of_3_edgetpu.tflite', '3148d81cdc86d50c62368afd9f882df0'), ('dumb_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_1_of_3_edgetpu.tflite', '6bb6a6bbaa604020deb9c1eb8e545c8a'), ('dumb_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_segment_2_of_3_edgetpu.tflite', 'cc4b255b2e50189d25c65867a4fadd6b')],
             '_tflite': ('all_segments_tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite', '17d1ee62d6975099ba957ed5f7472ced')
             },
             'efficientdet_lite0_320_ptq': {
#  23.2 ms/inference ( 43.1 FPS;  4.0 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite0_320_ptq
#  11.9 ms/inference ( 84.0 FPS;  7.8 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   8.3 ms/inference (119.9 FPS; 11.2 tensor MPx/sec) for 3 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   6.5 ms/inference (153.6 FPS; 14.3 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   6.0 ms/inference (166.7 FPS; 15.5 tensor MPx/sec) for 5 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   5.5 ms/inference (181.2 FPS; 16.9 tensor MPx/sec) for 6 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   5.3 ms/inference (188.0 FPS; 17.5 tensor MPx/sec) for 7 TPUs using 1 segments: efficientdet_lite0_320_ptq
#   4.9 ms/inference (203.3 FPS; 18.9 tensor MPx/sec) for 8 TPUs using 1 segments: efficientdet_lite0_320_ptq
             '_tflite': ('all_segments_efficientdet_lite0_320_ptq_edgetpu.tflite', '6e4e281e51b5f8b4ca335c32ac86b072')
             },
             'efficientdet_lite1_384_ptq': {
#  34.6 ms/inference ( 28.9 FPS;  3.9 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  17.7 ms/inference ( 56.4 FPS;  7.7 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  12.3 ms/inference ( 81.6 FPS; 11.1 tensor MPx/sec) for 3 TPUs using 1 segments: efficientdet_lite1_384_ptq
#  10.1 ms/inference ( 99.3 FPS; 13.5 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite1_384_ptq
#   8.7 ms/inference (114.9 FPS; 15.7 tensor MPx/sec) for 5 TPUs using 1 segments: efficientdet_lite1_384_ptq
#   7.7 ms/inference (130.5 FPS; 17.8 tensor MPx/sec) for 6 TPUs using 1 segments: efficientdet_lite1_384_ptq
#   7.2 ms/inference (138.9 FPS; 18.9 tensor MPx/sec) for 7 TPUs using 1 segments: efficientdet_lite1_384_ptq
#   6.8 ms/inference (146.0 FPS; 19.9 tensor MPx/sec) for 8 TPUs using 1 segments: efficientdet_lite1_384_ptq
             '_tflite': ('all_segments_efficientdet_lite1_384_ptq_edgetpu.tflite', 'cba6ce06f67d94bb388e26786356f99f')
             },
             'efficientdet_lite2_448_ptq': {
#  59.8 ms/inference ( 16.7 FPS;  3.1 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite2_448_ptq
#  30.4 ms/inference ( 32.9 FPS;  6.2 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite2_448_ptq
#  19.3 ms/inference ( 51.8 FPS;  9.7 tensor MPx/sec) for 3 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  15.9 ms/inference ( 62.8 FPS; 11.8 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite2_448_ptq
#  13.7 ms/inference ( 73.2 FPS; 13.7 tensor MPx/sec) for 5 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  11.7 ms/inference ( 85.3 FPS; 16.0 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite2_448_ptq
#  10.3 ms/inference ( 96.7 FPS; 18.1 tensor MPx/sec) for 7 TPUs using 2 segments: efficientdet_lite2_448_ptq
#   9.2 ms/inference (108.1 FPS; 20.3 tensor MPx/sec) for 8 TPUs using 2 segments: efficientdet_lite2_448_ptq
3: [('166x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', 'a1ba35d13759804521475106aebdb778'), ('166x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite', 'd4397a5fefe80b63b26a966f05bfbc0a')],
5: [('133x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '03dc0f13ab3588cce6d2a83949e1d05d'), ('133x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite', 'ea708ee1c538b5d9dd6c21c2397b2a95')],
6: [('2x_last_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', '7a28effcf51b0aa8708820a7a8e34bba'), ('2x_last_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite', 'd2411db586cc7c0add434a864ba7b25e')],
7: [('166x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', 'a1ba35d13759804521475106aebdb778'), ('166x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite', 'd4397a5fefe80b63b26a966f05bfbc0a')],
8: [('166x_first_seg_efficientdet_lite2_448_ptq_segment_0_of_2_edgetpu.tflite', 'a1ba35d13759804521475106aebdb778'), ('166x_first_seg_efficientdet_lite2_448_ptq_segment_1_of_2_edgetpu.tflite', 'd4397a5fefe80b63b26a966f05bfbc0a')],
             '_tflite': ('all_segments_efficientdet_lite2_448_ptq_edgetpu.tflite', 'a91e70b4f4785551c5c03791c284ddb0')
             },
             'efficientdet_lite3_512_ptq': {
#  75.3 ms/inference ( 13.3 FPS;  3.3 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  38.3 ms/inference ( 26.1 FPS;  6.5 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  26.0 ms/inference ( 38.5 FPS;  9.5 tensor MPx/sec) for 3 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  19.9 ms/inference ( 50.2 FPS; 12.4 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  16.6 ms/inference ( 60.1 FPS; 14.8 tensor MPx/sec) for 5 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  14.0 ms/inference ( 71.4 FPS; 17.6 tensor MPx/sec) for 6 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  12.8 ms/inference ( 78.4 FPS; 19.4 tensor MPx/sec) for 7 TPUs using 1 segments: efficientdet_lite3_512_ptq
#  11.7 ms/inference ( 85.3 FPS; 21.1 tensor MPx/sec) for 8 TPUs using 1 segments: efficientdet_lite3_512_ptq
             '_tflite': ('all_segments_efficientdet_lite3_512_ptq_edgetpu.tflite', '1e17272603f34514da09472f92272be3')
             },
             'efficientdet_lite3x_640_ptq': {
# 182.4 ms/inference (  5.5 FPS;  2.1 tensor MPx/sec) for 1 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  91.9 ms/inference ( 10.9 FPS;  4.2 tensor MPx/sec) for 2 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  61.8 ms/inference ( 16.2 FPS;  6.3 tensor MPx/sec) for 3 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  47.5 ms/inference ( 21.0 FPS;  8.2 tensor MPx/sec) for 4 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  38.7 ms/inference ( 25.9 FPS; 10.1 tensor MPx/sec) for 5 TPUs using 1 segments: efficientdet_lite3x_640_ptq
#  32.2 ms/inference ( 31.0 FPS; 12.1 tensor MPx/sec) for 6 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  28.4 ms/inference ( 35.2 FPS; 13.8 tensor MPx/sec) for 7 TPUs using 2 segments: efficientdet_lite3x_640_ptq
#  26.4 ms/inference ( 37.9 FPS; 14.8 tensor MPx/sec) for 8 TPUs using 2 segments: efficientdet_lite3x_640_ptq
3: [('2x_last_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', 'bd050997882668f4915f2a6df3267531'), ('2x_last_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite', '959862e6afcc306010ce67fdeda15375')],
6: [('all_segments_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '0c4e89b488f5cff8b29d0de23d231f0d'), ('all_segments_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite', 'd7a5a707e0a2b4410dcdc3b5d9d875f0')],
7: [('15x_first_seg_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '1aee305ef779b43f3b0ffb37b2ac98dd'), ('15x_first_seg_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite', '8eac175bd570f3b5411af3f4368b88de')],
8: [('all_segments_efficientdet_lite3x_640_ptq_segment_0_of_2_edgetpu.tflite', '0c4e89b488f5cff8b29d0de23d231f0d'), ('all_segments_efficientdet_lite3x_640_ptq_segment_1_of_2_edgetpu.tflite', 'd7a5a707e0a2b4410dcdc3b5d9d875f0')],
             '_tflite': ('all_segments_efficientdet_lite3x_640_ptq_edgetpu.tflite', 'a79af4afcf144002cc5b82691e3938f9')
             },
             'yolov5n-int8': {
#  26.5 ms/inference ( 37.7 FPS;  7.1 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5n-int8
#  13.8 ms/inference ( 72.5 FPS; 13.6 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5n-int8
#  10.2 ms/inference ( 98.2 FPS; 18.4 tensor MPx/sec) for 3 TPUs using 1 segments: yolov5n-int8
#   8.2 ms/inference (121.2 FPS; 22.7 tensor MPx/sec) for 4 TPUs using 1 segments: yolov5n-int8
#   7.4 ms/inference (135.0 FPS; 25.3 tensor MPx/sec) for 5 TPUs using 1 segments: yolov5n-int8
#   6.3 ms/inference (159.2 FPS; 29.8 tensor MPx/sec) for 6 TPUs using 1 segments: yolov5n-int8
#   6.3 ms/inference (159.2 FPS; 29.8 tensor MPx/sec) for 7 TPUs using 1 segments: yolov5n-int8
#   6.1 ms/inference (163.4 FPS; 30.6 tensor MPx/sec) for 8 TPUs using 2 segments: yolov5n-int8
8: [('all_segments_yolov5n-int8_segment_0_of_2_edgetpu.tflite', 'bdd58b4bfeccddf533d439643869083d'), ('all_segments_yolov5n-int8_segment_1_of_2_edgetpu.tflite', '076e13c75bf164ba91d747c615386a63')],
             '_tflite': ('all_segments_yolov5n-int8_edgetpu.tflite', 'dcd616c860324d1e653d4e32cf6ebea1')
             },
             'yolov5s-int8': {
#  37.1 ms/inference ( 27.0 FPS;  5.0 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5s-int8
#  18.8 ms/inference ( 53.1 FPS; 10.0 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5s-int8
#  13.0 ms/inference ( 77.0 FPS; 14.4 tensor MPx/sec) for 3 TPUs using 1 segments: yolov5s-int8
#  10.5 ms/inference ( 95.3 FPS; 17.9 tensor MPx/sec) for 4 TPUs using 1 segments: yolov5s-int8
#   8.5 ms/inference (117.6 FPS; 22.1 tensor MPx/sec) for 5 TPUs using 1 segments: yolov5s-int8
#   7.2 ms/inference (138.9 FPS; 26.1 tensor MPx/sec) for 6 TPUs using 1 segments: yolov5s-int8
#   6.7 ms/inference (149.0 FPS; 27.9 tensor MPx/sec) for 7 TPUs using 1 segments: yolov5s-int8
#   6.6 ms/inference (152.4 FPS; 28.6 tensor MPx/sec) for 8 TPUs using 1 segments: yolov5s-int8
             '_tflite': ('all_segments_yolov5s-int8_edgetpu.tflite', '05ade3a89b783c3118a76a4370c1a3b5')
             },
             'yolov5m-int8': {
# 100.7 ms/inference (  9.9 FPS;  1.9 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5m-int8
#  50.7 ms/inference ( 19.7 FPS;  3.7 tensor MPx/sec) for 2 TPUs using 1 segments: yolov5m-int8
#  31.2 ms/inference ( 32.1 FPS;  6.0 tensor MPx/sec) for 3 TPUs using 2 segments: yolov5m-int8
#  25.6 ms/inference ( 39.1 FPS;  7.3 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5m-int8
#  19.4 ms/inference ( 51.4 FPS;  9.7 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5m-int8
#  15.9 ms/inference ( 62.8 FPS; 11.8 tensor MPx/sec) for 6 TPUs using 3 segments: yolov5m-int8
#  14.4 ms/inference ( 69.6 FPS; 13.0 tensor MPx/sec) for 7 TPUs using 2 segments: yolov5m-int8
#  14.0 ms/inference ( 71.5 FPS; 13.4 tensor MPx/sec) for 8 TPUs using 2 segments: yolov5m-int8
3: [('15x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', 'e10855818f6265fa86b123a8a5442107'), ('15x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite', 'eb1dac2880acc20338306fe0799ba3ab')],
4: [('166x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '978fd75ec3c100d795de81e1637466a0'), ('166x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite', 'ddee56fa1804b857731cbf065ccefe1f')],
5: [('4x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '30649f7906a2196f30ba279201eeb892'), ('4x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite', 'c57b1cb5d50bac71c42f80cfb9064e6d')],
6: [('166x_first_seg_yolov5m-int8_segment_0_of_3_edgetpu.tflite', '20e354cfa4f2c5c0ce14d929e97a6ba4'), ('166x_first_seg_yolov5m-int8_segment_1_of_3_edgetpu.tflite', 'b8fb00331d6952510a38ac11c546b5a4'), ('166x_first_seg_yolov5m-int8_segment_2_of_3_edgetpu.tflite', 'c85782e9e46bb33d8d663183e9f3cb39')],
7: [('4x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '30649f7906a2196f30ba279201eeb892'), ('4x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite', 'c57b1cb5d50bac71c42f80cfb9064e6d')],
8: [('166x_first_seg_yolov5m-int8_segment_0_of_2_edgetpu.tflite', '978fd75ec3c100d795de81e1637466a0'), ('166x_first_seg_yolov5m-int8_segment_1_of_2_edgetpu.tflite', 'ddee56fa1804b857731cbf065ccefe1f')],
             '_tflite': ('all_segments_yolov5m-int8_edgetpu.tflite', '0c7ee152856677f94e7cd9ac507c3922')
             },
             'yolov5l-int8': {
# 182.9 ms/inference (  5.5 FPS;  1.0 tensor MPx/sec) for 1 TPUs using 1 segments: yolov5l-int8
#  84.9 ms/inference ( 11.8 FPS;  2.2 tensor MPx/sec) for 2 TPUs using 2 segments: yolov5l-int8
#  55.0 ms/inference ( 18.2 FPS;  3.4 tensor MPx/sec) for 3 TPUs using 3 segments: yolov5l-int8
#  43.3 ms/inference ( 23.1 FPS;  4.3 tensor MPx/sec) for 4 TPUs using 2 segments: yolov5l-int8
#  33.2 ms/inference ( 30.2 FPS;  5.7 tensor MPx/sec) for 5 TPUs using 2 segments: yolov5l-int8
#  29.3 ms/inference ( 34.1 FPS;  6.4 tensor MPx/sec) for 6 TPUs using 3 segments: yolov5l-int8
#  25.8 ms/inference ( 38.7 FPS;  7.3 tensor MPx/sec) for 7 TPUs using 6 segments: yolov5l-int8
#  20.8 ms/inference ( 48.0 FPS;  9.0 tensor MPx/sec) for 8 TPUs using 5 segments: yolov5l-int8
2: [('dumb_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '14ff8889dc057808c706b7a9a27d89f7'), ('dumb_yolov5l-int8_segment_1_of_2_edgetpu.tflite', '57d11f35ee8bb0bc019cdd321f8590be')],
3: [('all_segments_yolov5l-int8_segment_0_of_3_edgetpu.tflite', '5bf908acc1b4e71316b90b14714c4575'), ('all_segments_yolov5l-int8_segment_1_of_3_edgetpu.tflite', '346a1cdaacec7caeaee768cdcf0dcc36'), ('all_segments_yolov5l-int8_segment_2_of_3_edgetpu.tflite', '2a879b9104fc2a10bab85fb6f0805cb7')],
4: [('3x_first_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '107766470c5165067fd4055edfcdb18c'), ('3x_first_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite', 'fd8a6c1f4e3a3dc0bc91d36745e89520')],
5: [('3x_first_seg_yolov5l-int8_segment_0_of_2_edgetpu.tflite', '107766470c5165067fd4055edfcdb18c'), ('3x_first_seg_yolov5l-int8_segment_1_of_2_edgetpu.tflite', 'fd8a6c1f4e3a3dc0bc91d36745e89520')],
6: [('all_segments_yolov5l-int8_segment_0_of_3_edgetpu.tflite', '5bf908acc1b4e71316b90b14714c4575'), ('all_segments_yolov5l-int8_segment_1_of_3_edgetpu.tflite', '346a1cdaacec7caeaee768cdcf0dcc36'), ('all_segments_yolov5l-int8_segment_2_of_3_edgetpu.tflite', '2a879b9104fc2a10bab85fb6f0805cb7')],
7: [('dumb_yolov5l-int8_segment_0_of_6_edgetpu.tflite', '790b1a914dcf4279c7529670953b8707'), ('dumb_yolov5l-int8_segment_1_of_6_edgetpu.tflite', '501b3556acf1c9ca433b8d952820ea82'), ('dumb_yolov5l-int8_segment_2_of_6_edgetpu.tflite', '871efae1313898b6bb3f5aa4060fb540'), ('dumb_yolov5l-int8_segment_3_of_6_edgetpu.tflite', '5e780f45bb79c929fe671e0a1c730e1a'), ('dumb_yolov5l-int8_segment_4_of_6_edgetpu.tflite', 'b3819a3d6952636b6c478eb531b36d87'), ('dumb_yolov5l-int8_segment_5_of_6_edgetpu.tflite', 'bbd5a9abadc9c254d22b403d53d5d6bc')],
8: [('dumb_yolov5l-int8_segment_0_of_5_edgetpu.tflite', '138febc49003cbf3637516d4a54ae43e'), ('dumb_yolov5l-int8_segment_1_of_5_edgetpu.tflite', '7fc233d166278ea21c03dad2f39d57c6'), ('dumb_yolov5l-int8_segment_2_of_5_edgetpu.tflite', '075cd583ed57286302235428be273557'), ('dumb_yolov5l-int8_segment_3_of_5_edgetpu.tflite', '9442d7c7f007278cb0a2f392968db267'), ('dumb_yolov5l-int8_segment_4_of_5_edgetpu.tflite', 'bb6320aa3bc7363518d5d544b335c8c9')],
             '_tflite': ('all_segments_yolov5l-int8_edgetpu.tflite', '947a3ee13a2e6fca0ed52c6108413353')
             },
             'yolov8n_416_640px': {
#  23.6 ms/inference ( 42.3 FPS;  9.8 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8n_384_640px
#  11.9 ms/inference ( 84.1 FPS; 19.4 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8n_384_640px
#   8.1 ms/inference (123.2 FPS; 28.4 tensor MPx/sec) for 3 TPUs using 1 segments: yolov8n_384_640px
#   6.4 ms/inference (156.2 FPS; 36.0 tensor MPx/sec) for 4 TPUs using 1 segments: yolov8n_384_640px
#   4.8 ms/inference (210.1 FPS; 42.0 tensor MPx/sec) for 5 TPUs using 1 segments: yolov8n_352_608px
#   4.7 ms/inference (212.8 FPS; 49.1 tensor MPx/sec) for 6 TPUs using 1 segments: yolov8n_384_640px
#   4.7 ms/inference (212.8 FPS; 49.1 tensor MPx/sec) for 7 TPUs using 1 segments: yolov8n_384_640px
#   4.9 ms/inference (204.1 FPS; 51.1 tensor MPx/sec) for 8 TPUs using 3 segments: yolov8n_416_640px
8: [('3x_first_seg_yolov8n_416_640px_segment_0_of_3_edgetpu.tflite', '949cc2bf3815d5d870aeec595181541e'), ('3x_first_seg_yolov8n_416_640px_segment_1_of_3_edgetpu.tflite', 'd8fc08a86a1a82c420ed33c48f82d3af'), ('3x_first_seg_yolov8n_416_640px_segment_2_of_3_edgetpu.tflite', '31bdde82f94b4554012d6ecb7c201db1')],
             '_tflite': ('all_segments_yolov8n_384_640px_edgetpu.tflite', 'dbc5eb9f775b8cec51492ed80454b154')
             },
             'yolov8s_416_640px': {
#  46.7 ms/inference ( 21.4 FPS;  4.7 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8s_384_608px
#  23.5 ms/inference ( 42.5 FPS;  9.3 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8s_384_608px
#  15.9 ms/inference ( 63.1 FPS; 13.8 tensor MPx/sec) for 3 TPUs using 1 segments: yolov8s_384_608px
#  10.7 ms/inference ( 93.5 FPS; 18.7 tensor MPx/sec) for 4 TPUs using 2 segments: yolov8s_352_608px
#   9.4 ms/inference (106.8 FPS; 23.4 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8s_384_608px
#   8.1 ms/inference (124.2 FPS; 27.2 tensor MPx/sec) for 6 TPUs using 2 segments: yolov8s_384_608px
#   6.9 ms/inference (144.7 FPS; 31.7 tensor MPx/sec) for 7 TPUs using 2 segments: yolov8s_384_608px
#   5.6 ms/inference (177.3 FPS; 35.5 tensor MPx/sec) for 8 TPUs using 2 segments: yolov8s_352_608px
4: [('4x_first_seg_yolov8s_352_608px_segment_0_of_2_edgetpu.tflite', '6ecc47f6d9ef6b2a45ff4393105fe310'), ('4x_first_seg_yolov8s_352_608px_segment_1_of_2_edgetpu.tflite', 'a14cc49567acdc15d27173e3410b1e1e')],
5: [('3x_first_seg_yolov8s_384_608px_segment_0_of_2_edgetpu.tflite', '4b509292971a9a1da8da62f0b59fa8d4'), ('3x_first_seg_yolov8s_384_608px_segment_1_of_2_edgetpu.tflite', '4d95f76fbf94402b992bdd14b8b0ab09')],
6: [('3x_first_seg_yolov8s_384_608px_segment_0_of_2_edgetpu.tflite', '4b509292971a9a1da8da62f0b59fa8d4'), ('3x_first_seg_yolov8s_384_608px_segment_1_of_2_edgetpu.tflite', '4d95f76fbf94402b992bdd14b8b0ab09')],
7: [('2x_first_seg_yolov8s_384_608px_segment_0_of_2_edgetpu.tflite', '61057df9063a0940c61fb80d44c87d07'), ('2x_first_seg_yolov8s_384_608px_segment_1_of_2_edgetpu.tflite', '35286b2059e4a9cdbd0964f653476c97')],
8: [('4x_first_seg_yolov8s_352_608px_segment_0_of_2_edgetpu.tflite', '6ecc47f6d9ef6b2a45ff4393105fe310'), ('4x_first_seg_yolov8s_352_608px_segment_1_of_2_edgetpu.tflite', 'a14cc49567acdc15d27173e3410b1e1e')],
             '_tflite': ('all_segments_yolov8s_384_608px_edgetpu.tflite', 'a976571aa7b851c01934f19fbd954837')
             },
             'yolov8m_416_640px': {
# 187.6 ms/inference (  5.3 FPS;  1.1 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8m_352_608px
#  94.8 ms/inference ( 10.6 FPS;  2.6 tensor MPx/sec) for 2 TPUs using 2 segments: yolov8m_416_640px
#  57.2 ms/inference ( 17.5 FPS;  4.4 tensor MPx/sec) for 3 TPUs using 2 segments: yolov8m_416_640px
#  43.7 ms/inference ( 22.9 FPS;  5.7 tensor MPx/sec) for 4 TPUs using 2 segments: yolov8m_416_640px
#  35.1 ms/inference ( 28.5 FPS;  7.1 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8m_416_640px
#  30.6 ms/inference ( 32.7 FPS;  8.2 tensor MPx/sec) for 6 TPUs using 3 segments: yolov8m_416_640px
#  27.4 ms/inference ( 36.5 FPS;  9.2 tensor MPx/sec) for 7 TPUs using 4 segments: yolov8m_416_640px
#  26.6 ms/inference ( 37.6 FPS;  9.4 tensor MPx/sec) for 8 TPUs using 3 segments: yolov8m_416_640px
2: [('all_segments_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '2538c274186d50fd00f9135029036efe'), ('all_segments_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite', 'f70e3a43ae069e4499b1556aa26ab979')],
3: [('2x_last_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '645b50901151530a030c941067896619'), ('2x_last_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite', '25a69dbf0da3af9abb6150a2f1b8a2ae')],
4: [('2x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '91bc0025498094926aca5248dfa30998'), ('2x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite', 'e24331572eb782077841467c48b16d39')],
5: [('3x_first_seg_yolov8m_416_640px_segment_0_of_2_edgetpu.tflite', '06b5de52b30d69241c3e5522a87cbacb'), ('3x_first_seg_yolov8m_416_640px_segment_1_of_2_edgetpu.tflite', '7aca2df7c03ff3d599904c97e6788bb5')],
6: [('all_segments_yolov8m_416_640px_segment_0_of_3_edgetpu.tflite', '44553d512aa793ede73b027a0c489af4'), ('all_segments_yolov8m_416_640px_segment_1_of_3_edgetpu.tflite', '2c4cdf70060224436e4d9cef5f6d0096'), ('all_segments_yolov8m_416_640px_segment_2_of_3_edgetpu.tflite', '2d79b825a3eaee20fc0c43c200f5f988')],
7: [('all_segments_yolov8m_416_640px_segment_0_of_4_edgetpu.tflite', '6ccca16e93be61d56a04f9538fb21190'), ('all_segments_yolov8m_416_640px_segment_1_of_4_edgetpu.tflite', 'c58f17a741d30f77f78dc6c1bc0ecdee'), ('all_segments_yolov8m_416_640px_segment_2_of_4_edgetpu.tflite', '7d428ca816620e36965d3b442bd8aa27'), ('all_segments_yolov8m_416_640px_segment_3_of_4_edgetpu.tflite', 'a3b1ecbe3e47fbb0ae7843a387f8b66a')],
8: [('all_segments_yolov8m_416_640px_segment_0_of_3_edgetpu.tflite', '44553d512aa793ede73b027a0c489af4'), ('all_segments_yolov8m_416_640px_segment_1_of_3_edgetpu.tflite', '2c4cdf70060224436e4d9cef5f6d0096'), ('all_segments_yolov8m_416_640px_segment_2_of_3_edgetpu.tflite', '2d79b825a3eaee20fc0c43c200f5f988')],
             '_tflite': ('all_segments_yolov8m_352_608px_edgetpu.tflite', 'a59142cec40d2699898bbd82e90f67d2')
             },
             'yolov8l_416_640px': {
# 234.7 ms/inference (  4.3 FPS;  0.8 tensor MPx/sec) for 1 TPUs using 1 segments: yolov8l_352_608px
# 117.7 ms/inference (  8.5 FPS;  1.7 tensor MPx/sec) for 2 TPUs using 1 segments: yolov8l_352_608px
#  84.0 ms/inference ( 11.9 FPS;  2.6 tensor MPx/sec) for 3 TPUs using 2 segments: yolov8l_384_608px
#  58.9 ms/inference ( 17.0 FPS;  3.4 tensor MPx/sec) for 4 TPUs using 1 segments: yolov8l_352_608px
#  46.0 ms/inference ( 21.7 FPS;  4.3 tensor MPx/sec) for 5 TPUs using 2 segments: yolov8l_352_608px
#  50.6 ms/inference ( 19.7 FPS;  5.0 tensor MPx/sec) for 6 TPUs using 3 segments: yolov8l_416_640px
#  49.2 ms/inference ( 20.3 FPS;  5.1 tensor MPx/sec) for 7 TPUs using 2 segments: yolov8l_416_640px
#  44.6 ms/inference ( 22.4 FPS;  5.6 tensor MPx/sec) for 8 TPUs using 2 segments: yolov8l_416_640px
3: [('2x_first_seg_yolov8l_384_608px_segment_0_of_2_edgetpu.tflite', 'b0c271cc7593b1f9e80af31f00a2ca99'), ('2x_first_seg_yolov8l_384_608px_segment_1_of_2_edgetpu.tflite', '389adfc3a5b8a2fc2bbb45a3d81b8ef5')],
5: [('3x_first_seg_yolov8l_352_608px_segment_0_of_2_edgetpu.tflite', '35aaed3e61e1107767aee408f3b89865'), ('3x_first_seg_yolov8l_352_608px_segment_1_of_2_edgetpu.tflite', '1bf53847af16c9ae9a9b225020742610')],
6: [('4x_first_seg_yolov8l_416_640px_segment_0_of_3_edgetpu.tflite', '146f8cd733e3ed03ccb0b01f1b2f6e73'), ('4x_first_seg_yolov8l_416_640px_segment_1_of_3_edgetpu.tflite', '0c3ebe468b46142aae8d6a358cde97dd'), ('4x_first_seg_yolov8l_416_640px_segment_2_of_3_edgetpu.tflite', '694c46bc795510235355efcc23f50b3d')],
7: [('2x_first_seg_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', '3ac992275b06e8d32f5a84daec41889f'), ('2x_first_seg_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite', '2b103ec1a0f9812dd9545a94fbd2de9b')],
8: [('15x_first_seg_yolov8l_416_640px_segment_0_of_2_edgetpu.tflite', 'b9b49ba86b453a5e3365535b241245b3'), ('15x_first_seg_yolov8l_416_640px_segment_1_of_2_edgetpu.tflite', '1f81c69cda86b22244bf83599b77274a')],
             '_tflite': ('all_segments_yolov8l_352_608px_edgetpu.tflite', 'f4e5f929370752ade565d5b048b214a7')
             },
             'yolov9t_416_640px': {
#  28.7 ms/inference ( 34.9 FPS;  8.0 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9t_384_640px
#  14.5 ms/inference ( 69.1 FPS; 15.9 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9t_384_640px
#   9.8 ms/inference (101.7 FPS; 23.5 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9t_384_640px
#   7.6 ms/inference (131.8 FPS; 30.4 tensor MPx/sec) for 4 TPUs using 1 segments: yolov9t_384_640px
#   6.3 ms/inference (158.5 FPS; 36.5 tensor MPx/sec) for 5 TPUs using 1 segments: yolov9t_384_640px
#   5.3 ms/inference (190.1 FPS; 43.9 tensor MPx/sec) for 6 TPUs using 1 segments: yolov9t_384_640px
#   4.6 ms/inference (217.9 FPS; 50.3 tensor MPx/sec) for 7 TPUs using 1 segments: yolov9t_384_640px
#   4.5 ms/inference (223.7 FPS; 51.6 tensor MPx/sec) for 8 TPUs using 1 segments: yolov9t_384_640px
             '_tflite': ('all_segments_yolov9t_384_640px_edgetpu.tflite', '95f95ee11fd25ade78ebc70168a1ec3f')
             },
             'yolov9s_416_640px': {
#  45.1 ms/inference ( 22.2 FPS;  4.2 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9s_352_576px
#  22.7 ms/inference ( 44.1 FPS;  8.3 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9s_352_576px
#  15.3 ms/inference ( 65.5 FPS; 12.4 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9s_352_576px
#  11.5 ms/inference ( 87.0 FPS; 16.5 tensor MPx/sec) for 4 TPUs using 1 segments: yolov9s_352_576px
#   9.3 ms/inference (107.2 FPS; 20.3 tensor MPx/sec) for 5 TPUs using 1 segments: yolov9s_352_576px
#   8.0 ms/inference (125.6 FPS; 23.8 tensor MPx/sec) for 6 TPUs using 1 segments: yolov9s_352_576px
#   6.9 ms/inference (144.3 FPS; 27.3 tensor MPx/sec) for 7 TPUs using 1 segments: yolov9s_352_576px
#   6.2 ms/inference (161.8 FPS; 30.6 tensor MPx/sec) for 8 TPUs using 1 segments: yolov9s_352_576px
             '_tflite': ('all_segments_yolov9s_352_576px_edgetpu.tflite', 'da983c998888f7f152dd59c4001e6465')
             },
             'yolov9m_416_640px': {
# 146.9 ms/inference (  6.8 FPS;  1.3 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9m_352_576px
#  73.6 ms/inference ( 13.6 FPS;  2.6 tensor MPx/sec) for 2 TPUs using 1 segments: yolov9m_352_576px
#  46.9 ms/inference ( 21.3 FPS;  4.0 tensor MPx/sec) for 3 TPUs using 2 segments: yolov9m_352_576px
#  35.4 ms/inference ( 28.2 FPS;  5.3 tensor MPx/sec) for 4 TPUs using 2 segments: yolov9m_352_576px
#  29.5 ms/inference ( 33.9 FPS;  6.4 tensor MPx/sec) for 5 TPUs using 2 segments: yolov9m_352_576px
#  33.2 ms/inference ( 30.1 FPS;  7.6 tensor MPx/sec) for 6 TPUs using 2 segments: yolov9m_416_640px
#  21.9 ms/inference ( 45.7 FPS;  8.6 tensor MPx/sec) for 7 TPUs using 2 segments: yolov9m_352_576px
#  19.8 ms/inference ( 50.4 FPS;  9.5 tensor MPx/sec) for 8 TPUs using 2 segments: yolov9m_352_576px
3: [('15x_first_seg_yolov9m_352_576px_segment_0_of_2_edgetpu.tflite', '5821e54b26d69e4cb1c2d2353b2a32b8'), ('15x_first_seg_yolov9m_352_576px_segment_1_of_2_edgetpu.tflite', 'af1a3b119110c2f13f19c8e9ed39a348')],
4: [('2x_first_seg_yolov9m_352_576px_segment_0_of_2_edgetpu.tflite', 'ca8e62206d0998521cdcdf482a8aafa7'), ('2x_first_seg_yolov9m_352_576px_segment_1_of_2_edgetpu.tflite', '8722298ca656a9aa905b47eebd99f143')],
5: [('4x_first_seg_yolov9m_352_576px_segment_0_of_2_edgetpu.tflite', 'd13d791a033421664d314cd8fe7159a5'), ('4x_first_seg_yolov9m_352_576px_segment_1_of_2_edgetpu.tflite', '23f0abd7c8fc2189afe824307c7e764e')],
6: [('4x_first_seg_yolov9m_416_640px_segment_0_of_2_edgetpu.tflite', '3575d250197920b946d9a007e3175138'), ('4x_first_seg_yolov9m_416_640px_segment_1_of_2_edgetpu.tflite', '28d49a444bdf477f001bf084adcd2d65')],
7: [('3x_first_seg_yolov9m_352_576px_segment_0_of_2_edgetpu.tflite', 'c037de1514a795711ff2fcf9207a9d9d'), ('3x_first_seg_yolov9m_352_576px_segment_1_of_2_edgetpu.tflite', 'fa2f5522c0244aaab5be4ba03e829319')],
8: [('4x_first_seg_yolov9m_352_576px_segment_0_of_2_edgetpu.tflite', 'd13d791a033421664d314cd8fe7159a5'), ('4x_first_seg_yolov9m_352_576px_segment_1_of_2_edgetpu.tflite', '23f0abd7c8fc2189afe824307c7e764e')],
             '_tflite': ('all_segments_yolov9m_352_576px_edgetpu.tflite', '1bdb7e615eb275b74f0791448011a70f')
             },
             'yolov9c_416_640px': {
# 304.4 ms/inference (  3.3 FPS;  0.8 tensor MPx/sec) for 1 TPUs using 1 segments: yolov9c_416_640px
# 112.2 ms/inference (  8.9 FPS;  1.7 tensor MPx/sec) for 2 TPUs using 2 segments: yolov9c_352_576px
# 102.0 ms/inference (  9.8 FPS;  2.5 tensor MPx/sec) for 3 TPUs using 1 segments: yolov9c_416_640px
#  72.6 ms/inference ( 13.8 FPS;  3.5 tensor MPx/sec) for 4 TPUs using 2 segments: yolov9c_416_640px
#  59.2 ms/inference ( 16.9 FPS;  4.2 tensor MPx/sec) for 5 TPUs using 2 segments: yolov9c_416_640px
#  38.6 ms/inference ( 25.9 FPS;  4.9 tensor MPx/sec) for 6 TPUs using 2 segments: yolov9c_352_576px
#  35.1 ms/inference ( 28.5 FPS;  5.4 tensor MPx/sec) for 7 TPUs using 2 segments: yolov9c_352_576px
#  32.8 ms/inference ( 30.5 FPS;  5.8 tensor MPx/sec) for 8 TPUs using 2 segments: yolov9c_352_576px
2: [('all_segments_yolov9c_352_576px_segment_0_of_2_edgetpu.tflite', '8c222f8b68ac3409c911e94d9fe238b5'), ('all_segments_yolov9c_352_576px_segment_1_of_2_edgetpu.tflite', '19cad259fa12078a9c578edc75237039')],
4: [('dumb_yolov9c_416_640px_segment_0_of_2_edgetpu.tflite', '9a12743271de03b8cbc418010ca49dd7'), ('dumb_yolov9c_416_640px_segment_1_of_2_edgetpu.tflite', '9c29968741d13145b452f0d5e78f72af')],
5: [('15x_last_seg_yolov9c_416_640px_segment_0_of_2_edgetpu.tflite', 'c3db29ffe0eea32d18d3f1ce00c67977'), ('15x_last_seg_yolov9c_416_640px_segment_1_of_2_edgetpu.tflite', '98798da3713530e359ff3045ddc02cfe')],
6: [('2x_last_seg_yolov9c_352_576px_segment_0_of_2_edgetpu.tflite', '6411b054b7e36a781b533a47314e1f43'), ('2x_last_seg_yolov9c_352_576px_segment_1_of_2_edgetpu.tflite', '5b74321e75daab50ec3708bd04dac846')],
7: [('15x_last_seg_yolov9c_352_576px_segment_0_of_2_edgetpu.tflite', '37b01ef791aae70d97877949c0b40421'), ('15x_last_seg_yolov9c_352_576px_segment_1_of_2_edgetpu.tflite', '3732f1586b0798ae6a078a13a79c0832')],
8: [('2x_last_seg_yolov9c_352_576px_segment_0_of_2_edgetpu.tflite', '6411b054b7e36a781b533a47314e1f43'), ('2x_last_seg_yolov9c_352_576px_segment_1_of_2_edgetpu.tflite', '5b74321e75daab50ec3708bd04dac846')],
             '_tflite': ('all_segments_yolov9c_416_640px_edgetpu.tflite', 'c02b9fec754b84c4e4c1757fd2122ddf')
             },
             'ipcam-general-v8': {
# 233.5 ms/inference (  4.3 FPS;  1.1 tensor MPx/sec) for 1 TPUs using 1 segments: ipcam-general-v8
#  44.6 ms/inference ( 22.4 FPS;  5.6 tensor MPx/sec) for 2 TPUs using 2 segments: ipcam-general-v8
#  22.4 ms/inference ( 44.6 FPS; 11.2 tensor MPx/sec) for 3 TPUs using 2 segments: ipcam-general-v8
#  16.0 ms/inference ( 62.5 FPS; 15.7 tensor MPx/sec) for 4 TPUs using 2 segments: ipcam-general-v8
#  12.1 ms/inference ( 82.9 FPS; 20.8 tensor MPx/sec) for 5 TPUs using 2 segments: ipcam-general-v8
#   9.9 ms/inference (101.1 FPS; 25.3 tensor MPx/sec) for 6 TPUs using 2 segments: ipcam-general-v8
#   8.9 ms/inference (112.7 FPS; 28.2 tensor MPx/sec) for 7 TPUs using 2 segments: ipcam-general-v8
#   8.7 ms/inference (114.7 FPS; 28.8 tensor MPx/sec) for 8 TPUs using 2 segments: ipcam-general-v8
2: [('all_segments_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', 'b9e472613162f8c16eb809ed57132741'), ('all_segments_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', 'b1ed179f61c50bfd97fbfbd3b3acacbc')],
3: [('166x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '075b6f8c13ff4df6be11f9fb5199f974'), ('166x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', '2f12424f4695b9b555a5fc7ffd6b8b97')],
4: [('2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '8c87ba2ac71450df22e0d4bfe6f2321e'), ('2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', '85b38d05ef94b39f409d7de7688ba053')],
5: [('2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '8c87ba2ac71450df22e0d4bfe6f2321e'), ('2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', '85b38d05ef94b39f409d7de7688ba053')],
6: [('2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '8c87ba2ac71450df22e0d4bfe6f2321e'), ('2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', '85b38d05ef94b39f409d7de7688ba053')],
7: [('4x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', 'ace0bc3dce916f0b739ee1f9350e6c0d'), ('4x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', 'a89dceb8ced5134078d15922ce77e560')],
8: [('2x_first_seg_ipcam-general-v8_segment_0_of_2_edgetpu.tflite', '8c87ba2ac71450df22e0d4bfe6f2321e'), ('2x_first_seg_ipcam-general-v8_segment_1_of_2_edgetpu.tflite', '85b38d05ef94b39f409d7de7688ba053')],
             '_tflite': ('all_segments_ipcam-general-v8_edgetpu.tflite', '89ded45c3c3d9d7a3e12100ccdd9627b')
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
