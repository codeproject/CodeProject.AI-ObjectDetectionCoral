# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```

Running this directly from src\runtimes\bin\windows\python37:

cd \src\runtimes\bin\windows\python37
python.exe coral\pycoral\examples\classify_image.py --model coral\pycoral\test_data\mobilenet_v2_1.0_224_inat_bird_quant.tflite --labels coral\pycoral\test_data\inat_bird_labels.txt --input coral\pycoral\test_data\parrot.jpg
"""
import argparse
import concurrent.futures
import copy
from datetime import datetime
import fnmatch
import logging
import os
import threading
import time
import cv2
#import tracemalloc

from PIL import Image
from PIL import ImageDraw

from options import Options
from tpu_runner import TPURunner, INTERPRETER_LIFESPAN_SECONDS

_tpu_runner = None
_last_model_check    = None  # When were the models last checked?
_model_list          = None
_model_list_lock     = threading.Lock()


def init_detect(options: Options, tpu_limit: int = -1) -> (str,str):
    global _tpu_runner

    _tpu_runner = TPURunner(tpu_limit = tpu_limit)
    _tpu_runner.max_idle_secs_before_recycle = options.max_idle_secs_before_recycle
    _tpu_runner.watchdog_idle_secs           = options.watchdog_idle_secs
    _tpu_runner.interpreter_lifespan_secs    = options.interpreter_lifespan_secs
    _tpu_runner.max_pipeline_queue_length    = options.max_pipeline_queue_length
    _tpu_runner.warn_temperature_thresh_C    = options.warn_temperature_thresh_C

    with _tpu_runner.runner_lock:
        return _tpu_runner.init_pipe(options)


def list_models(options:Options):
    
    # HACK: If we can't get the TPU interpreter created then let's fall back to
    #       the non-edge library TPU / TFLite-CPU code
    if not _tpu_runner or not _tpu_runner.pipeline_ok(options):
        logging.warning("No multi-TPU interpreters: Falling back to single-TPU/CPU listing")
        import objectdetection_coral_singletpu as odcs
        return odcs.list_models(options)

    global _last_model_check
    global _model_list

    supported_models = [ 'MobileNet SSD', 'EfficientDet-Lite', 'YOLOv5', 'YOLOv8' ] 

    # Check to make sure we aren't checking too often
    now_ts = datetime.now()
    if not _model_list or not _last_model_check or \
       (now_ts - _last_model_check).total_seconds() > 30:

        _last_model_check = now_ts

        with _model_list_lock:
          _model_list = []
          for model_name in supported_models:
              model_index = model_name.lower()
              pattern     = options.MODEL_SETTINGS[model_index][options.model_size].model_name_pattern
              for file in os.listdir(options.models_dir):
                  if fnmatch.fnmatch(file, '*' + pattern + '*'):
                      _model_list.append(model_name)
                      break

    return {
        "success": True,
        "models": _model_list
    }


def do_detect(options: Options, image: Image, score_threshold: float = 0.5):
    
    # HACK: If we can't get the TPU interpreter created then let's fall back to
    #       the non-edge library TPU / TFLite-CPU code
    if not _tpu_runner or not _tpu_runner.pipeline_ok(options):
        logging.warning("No multi-TPU interpreters: Falling back to single-TPU/CPU detection")
        import objectdetection_coral_singletpu as odcs
        if not odcs.interpreter_created:
            _, _ = odcs.init_detect(options)
        return odcs.do_detect(options, image, score_threshold)

    # Run inference
    predictions, inferenceMs, error = _tpu_runner.process_image(options, image, score_threshold)

    if not predictions:
        return {
            "success"     : False,
            "error"       : error,
            "count"       : 0,
            "predictions" : [],
            "inferenceMs" : 0
        }

    # Get output
    outputs = []
    for obj in predictions:
        class_id = obj.id
        caption  = _tpu_runner.labels.get(class_id, class_id)
        score    = float(obj.score)
        xmin, ymin, xmax, ymax = obj.bbox

        if score >= score_threshold:
            detection = {
                "confidence": score,
                "label": caption,
                "x_min": xmin,
                "y_min": ymin,
                "x_max": xmax,
                "y_max": ymax,
            }

            outputs.append(detection)

    return {
        "success"         : True,
        "count"           : len(outputs),
        "predictions"     : outputs,
        "inferenceMs"     : inferenceMs,
        "inferenceDevice" : _tpu_runner.device_type
    }


def cleanup():
  global _tpu_runner
  
  if _tpu_runner:
    _tpu_runner.__del__()
  _tpu_runner = None


# ------------------------------------------------------------------------------
# For Debug / command line calls

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True, nargs='+',
                      help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True,
                      help='File path of image to process')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-n', '--num-tpus', type=int, default=-1,
                      help='Restrict TPU count')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  parser.add_argument('-d', '--debug', action='store_true')
  args = parser.parse_args()

  if args.debug:
    logging.root.setLevel(logging.DEBUG)
  else:
    logging.root.setLevel(logging.INFO)

  options = Options()

  # Load segments
  if len(args.model) > 1:
    options.tpu_segments_lists = args.model
  else:
    options.model_cpu_file = args.model[0]
    options.model_tpu_file = args.model[0]

  # Limit to one tile
  # Allows us apples-to-apples comparisons when benchmarking
  options.downsample_by  = 100
  
  options.label_file = args.labels
  #image = Image.open(args.input)
  image = cv2.imread(args.input, cv2.IMREAD_COLOR)
  init_detect(options, args.num_tpus)

  print('----INFERENCE TIME----')
  print('Note: The first inference is slow because it includes',
        'loading the model into Edge TPU memory.')

  #tracemalloc.start()
  
  thread_cnt = 16
  tot_infr_time = 0 
  q_infr_count = 0
  q_wall_start = None

  half_wall_start = None
  half_infr_count = 0 
  start = time.perf_counter()

  if args.count > 1:
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_cnt) as executor:
      for chunk_i in range(0, args.count-1, thread_cnt*8):
        fs = [executor.submit(_tpu_runner.process_image, options, copy.copy(image), args.threshold)
              for i in range(min(thread_cnt*8, args.count-1 - chunk_i))]
        for f in concurrent.futures.as_completed(fs):
          _, infr_time, _ = f.result()
          tot_infr_time += infr_time

          # Start a timer for the last ~quarter of the run for more accurate benchmark
          if chunk_i > (args.count-1) * 3.0 / 4.0:
            q_infr_count += 1
            if q_wall_start is None:
              q_wall_start = time.perf_counter()
        
        # Uncomment for testing
        # import random
        # logging.info("Pause")
        # time.sleep(random.randint(0,INTERPRETER_LIFESPAN_SECONDS*3))
  
  # snapshot = tracemalloc.take_snapshot()
  # top_stats = snapshot.statistics('lineno')
  # for stat in top_stats[:20]:
  #   print(stat)

  start_one = time.perf_counter()
  objs, infr_time, _ = _tpu_runner.process_image(options, copy.copy(image), args.threshold)
  tot_infr_time += infr_time
  q_infr_count += 1
  wall_time = time.perf_counter() - start

  q_wall_time = 0.0
  mpps = 0.0
  if q_wall_start is not None:
    q_wall_time = time.perf_counter() - q_wall_start

    mpps = (_tpu_runner.input_details['shape'][1] - options.tile_overlap) \
           * (_tpu_runner.input_details['shape'][2] - options.tile_overlap) \
           * q_infr_count \
           / (q_wall_time * 1000000)
  
  logging.info('completed one run every %.2fms for %d runs; %.2fms wall time for a single run' %
                            (wall_time * 1000 / args.count, args.count,
                            (time.perf_counter() - start_one) * 1000))
                            
  logging.info('%.2fms avg time blocked across %d threads; %.2fms ea for final %d inferences; %.2f tensor MPx / sec' %
                            (tot_infr_time / args.count, thread_cnt,
                             q_wall_time * 1000 / q_infr_count, q_infr_count,
                             mpps))

  logging.info('-------RESULTS--------')
  if not objs:
    logging.info('No objects detected')
    return
  
  if any(objs):
    for obj in objs:
      logging.info(_tpu_runner.labels.get(obj.id, obj.id))
      logging.info(f'  id:    {obj.id}')
      logging.info(f'  score: {obj.score}')
      logging.info(f'  bbox:  {obj.bbox}')
  
  if args.output:
    # image = image.convert('RGB')
    image = Image.fromarray(image)
    draw_objects(ImageDraw.Draw(image), objs, _tpu_runner.labels)
    image.save(args.output, subsampling=2, quality=95)
    image.show()


if __name__ == '__main__':
  main()
  # Don't wait for watchdog during testing
  _tpu_runner.pipe.delete()
  os._exit(os.EX_OK)
  cleanup()
