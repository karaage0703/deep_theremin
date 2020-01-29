# -*- coding: utf-8 -*-
# Deep Theremin
import argparse
import cv2
import numpy as np
import os
import sys
import time
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import model_from_json
import serial
import pyrealsense2 as rs

from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Path to label and frozen detection graph. This is the actual model that is used for the object detection.
parser = argparse.ArgumentParser(description='Deep Theremin')
parser.add_argument('-m', '--model', default='./frozen_inference_graph.pb')
parser.add_argument('--save_image', action='store_true')

cascade_path = "./haarcascade_frontalface_alt.xml"

X_SIZE = 640
Y_SIZE = 480

SAVE_TIME_INTERVAL = 1

args = parser.parse_args()

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, X_SIZE, Y_SIZE, rs.format.z16, 60)
config.enable_stream(rs.stream.color, X_SIZE, Y_SIZE, rs.format.bgr8, 60)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

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

print('Loading recognition model...')
model_pred = model_from_json(open('mnist_deep_model.json').read())
model_pred.load_weights('weights.99.hdf5')
print('Model is loaded')

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


count_max = 0

if __name__ == '__main__':
  ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)
  count = 0

  base_freq = 50.0
  freq = base_freq

  pred_label = ''
  save_image_timer = time.time()

  try:
    while True:
      frames = pipeline.wait_for_frames()
      # frames.get_depth_frame() is a 640x360 depth image

      # Align the depth frame to color frame
      aligned_frames = align.process(frames)

      # Get aligned frames
      aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
      color_frame = aligned_frames.get_color_frame()

      # Validate that both frames are valid
      if not aligned_depth_frame or not color_frame:
          continue

      depth_image = np.asanyarray(aligned_depth_frame.get_data())
      color_image = np.asanyarray(color_frame.get_data())
      img = color_image

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

        box_size_max = 0.0
        speed_info = '%s' % ('speed: None')
        freq_info = '%s' % ('freq: None')

        for i in range(output_dict['num_detections']):
          detection_score = output_dict['detection_scores'][i]

          if detection_score > 0.80:
            # Define bounding box
            h, w, c = img.shape
            box = output_dict['detection_boxes'][i] * np.array( \
              [h, w,  h, w])
            box = box.astype(np.int)

            box_size = box[2] - box[0] + box[3] - box[1]
            if box_size_max < box_size:
              box_size_max = box_size

              center_pos = [(int)((box[0] + box[2]) / 2),
                              (int)((box[1] + box[3]) / 2)]
              distance = (1 / depth_image[center_pos[0], center_pos[1]]) * 1000

              freq = base_freq + 2 * distance * distance

              score_info = output_dict['detection_scores'][i]
              speed_info = '%s: %f' % ('speed=', elapsed_time)
              freq_info = '%s: %f' % ('freq=', freq)

              # crop hand
              hand_img = img[box[0]:box[2],box[1]:box[3]]

              # Prediction hand shape
              X = []
              hand_img = cv2.resize(hand_img, (64, 64))

              # Save Image
              if args.save_image:
                # elapsed_time = time.time() - save_image_timer
                if (time.time() - save_image_timer) > SAVE_TIME_INTERVAL:
                  save_image_timer = time.time()
                  image_file_name = datetime.datetime.today().strftime('%Y%m%d_%H%M%S') + '.jpg'

                  cv2.imwrite(image_file_name, hand_img)

              hand_img = img_to_array(hand_img)
              X.append(hand_img)
              X = np.asarray(X)
              X = X / 255.0
              start = time.time()
              preds = model_pred.predict(X)
              elapsed_time = time.time() - start

              labels = ['gu', 'choki', 'pa']
              label_num = 0
              tmp_max_pred = 0
              for i in preds[0]:
                if i > tmp_max_pred:
                  pred_label = labels[label_num]
                  tmp_max_pred = i
                label_num += 1

              # Draw bounding box
              cv2.rectangle(img, \
                (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 3)

              # Put label near bounding box
              # information = '%s: %f' % ('hand', output_dict['detection_scores'][i])
              information = '%s: %f' % ('hand', detection_score)
              cv2.putText(img, information, (box[1], box[2]), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)


        # Put info
        cv2.putText(img, speed_info, (10,50), \
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, freq_info, (10,100), \
          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        print(freq)
        # serial_txt = str(freq) + ',' + pred_label
        # ser.write(serial_txt.encode())
        ser.write(str(freq).encode())

        cv2.imshow('Deep Theremin', img)
        # cv2.imshow('Deep Theremin', image_np)
        count = 0

    tf_sess.close()
    cv2.destroyAllWindows()
  finally:
    pipeline.stop()
