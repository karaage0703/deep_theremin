# coding: utf-8
# Deep Theremin
import argparse
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf
import client

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='Deep Theremin')
parser.add_argument('-m', '--model', default='./frozen_inference_graph.pb')
parser.add_argument('-d', '--device', default='normal_cam') # normal_cam / jetson_nano_raspi_cam / jetson_nano_web_cam

cascade_path = "./haarcascade_frontalface_alt.xml"

args = parser.parse_args()

detection_graph = tf.Graph()

def load_graph():
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args.model, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# Load a (frozen) Tensorflow model into memory.
print('Loading graph...')
detection_graph = load_graph()
print('Graph is loaded')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with detection_graph.as_default():
  tf_sess = tf.Session(config = tf_config)
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)

  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

def run_inference_for_single_image(image, graph):
  # Run inference
  output_dict = tf_sess.run(tensor_dict,
                          feed_dict={image_tensor: image})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

# Switch camera according to device
if args.device == 'normal_cam':
  cam = cv2.VideoCapture(0)
elif args.device == 'jetson_nano_raspi_cam':
  GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'
  cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
elif args.device == 'jetson_nano_web_cam':
  cam = cv2.VideoCapture(1)
else:
  print('wrong device')
  sys.exit()

count_max = 0

if __name__ == '__main__':
  count = 0

  freq = 440/7

  while True:
    ret, img = cam.read()
    if not ret:
      print('error')
      break
    key = cv2.waitKey(1)
    if key == 27: # when ESC key is pressed break
        break

    count += 1
    if count > count_max:
      img_bgr = cv2.resize(img, (300, 300))

      # convert bgr to rgb
      image_np = img_bgr[:,:,::-1]

      # detect face and eliminate face region of image
      image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
      cascade = cv2.CascadeClassifier(cascade_path)
      facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

      if len(facerect) > 0:
        for rect in facerect:
          image_np[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 0.0

      image_np_expanded = np.expand_dims(image_np, axis=0)
      start = time.time()
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      elapsed_time = time.time() - start

      for i in range(output_dict['num_detections']):
        detection_score = output_dict['detection_scores'][i]

        if detection_score > 0.95:
          # Define bounding box
          h, w, c = img.shape
          box = output_dict['detection_boxes'][i] * np.array( \
            [h, w,  h, w])
          box = box.astype(np.int)

          distance = (output_dict['detection_boxes'][i][2] - output_dict['detection_boxes'][i][0]) \
                  + (output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1])

          freq = 440/7 +  (distance * 5) * (distance * 5)
          #with open('freq.txt', mode='w') as f:
          #  f.write(str(freq))
          client.send(freq)

          speed_info = '%s: %f' % ('speed=', elapsed_time)
          cv2.putText(img, speed_info, (10,50), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

          # Draw bounding box
          cv2.rectangle(img, \
            (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)

          # Put label near bounding box
          information = '%s: %f' % ('hand', output_dict['detection_scores'][i])
          cv2.putText(img, information, (box[1], box[2]), \
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

      cv2.imshow('Deep Theremin', img)
      # cv2.imshow('Deep Theremin', image_np)
      count = 0

  tf_sess.close()
  cam.release()
  cv2.destroyAllWindows()
