import re
import tensorflow as tf
import numpy as np
import math
from PIL import Image
from types import SimpleNamespace

style_predict_path = tf.keras.utils.get_file(
    'style_predict.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3/int8/predict/1?lite-format=tflite')
style_transform_path = style_transform_path = tf.keras.utils.get_file(
    'style_transform.tflite', 'https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3/int8/transfer/1?lite-format=tflite')

STYLE_SIZE = 256
CONTENT_SIZE = 384

def get_array_of_pieces(img, cfg):
  cols = cfg.cols
  rows = cfg.rows
  edge_size = cfg.edge_size

  pieces = []
  #print(title, "image size:", img.size)
  w = math.floor(img.size[0]/cols)
  h = math.floor(img.size[1]/rows)
  #print("using cell size:", w, "x", h)

  if (w < STYLE_SIZE or h < STYLE_SIZE):
    print("WARNING: That size seems a bit small and will probably result in stretching.")

  # scale edge_size
  we = w*edge_size/CONTENT_SIZE
  he = h*edge_size/CONTENT_SIZE

  for r in range(0, rows):
    for c in range(0, cols):
      el = 0 if c == 0 else we
      er = 0 if c == cols-1 else we
      eu = 0 if r == 0 else he
      ed = 0 if r == rows-1 else he

      region = img.crop((c*w-el, r*h-eu, (c+1)*w+er, (r+1)*h+ed))
      pieces.append(region)
  return pieces

def get_intermediate_tiles(img, cfg):
  cols = cfg.cols
  rows = cfg.rows
  edge_size = cfg.edge_size

  row_pieces = []
  column_pieces = []

  w = math.floor(img.size[0]/cols)
  h = math.floor(img.size[1]/rows)

  # scale edge_size
  we = w*edge_size/STYLE_SIZE
  he = h*edge_size/STYLE_SIZE

  for r in range(0, rows):
    for c in range(0, cols-1):
      el = 0 if c == 0 else we
      er = 0 if c == cols-1 else we
      eu = 0 if r == 0 else he
      ed = 0 if r == rows-1 else he

      region = img.crop(((c+0.5)*w-el, r*h-eu, (c+1.5)*w+er, (r+1)*h+ed))
      row_pieces.append(region)

  for r in range(0, rows-1):
    for c in range(0, cols):
      el = 0 if c == 0 else we
      er = 0 if c == cols-1 else we
      eu = 0 if r == 0 else he
      ed = 0 if r == rows-1 else he

      region = img.crop(((c)*w-el, (r+0.5)*h-eu, (c+1)*w+er, (r+1.5)*h+ed))
      column_pieces.append(region)

  return [row_pieces, column_pieces]

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]
  return img

# Function to load an image from a file, and add a batch dimension.
def load_content_img(image_pixels):
    if image_pixels.shape[-1] == 4:
        image_pixels = Image.fromarray(image_pixels)
        img = image_pixels.convert('RGB')
        img = np.array(img)
        img = tf.convert_to_tensor(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 3:
        img = tf.convert_to_tensor(image_pixels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :]
        return img
    elif image_pixels.shape[-1] == 1:
        raise Error(
            'Grayscale images not supported! Please try with RGB or RGBA images.')
    print('Exception not thrown')


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  # Calculate style bottleneck.
  interpreter.invoke()
  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
  )()
  # print('Style Bottleneck Shape:', style_bottleneck.shape)
  return style_bottleneck

# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image

def preprocessor(img, res):
  img = load_content_img(np.array(img))
  return preprocess_image(img, res)

def run_style_transform(style_bottleneck, preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()

  # Set model inputs.
  for index in range(len(input_details)):
    if input_details[index]["name"] == 'Conv/BiasAdd':
      interpreter.set_tensor(input_details[index]["index"], style_bottleneck)
    elif input_details[index]["name"] == 'content_image':
      interpreter.set_tensor(
          input_details[index]["index"], preprocessed_content_image)
  interpreter.invoke()

  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
  )()

  return stylized_image

def stylize(preprocessed_content_image, style_bottleneck, style_bottleneck_content, content_blending_ratio):
  # Blend the style bottleneck of style image and content image
  style_bottleneck_blended = content_blending_ratio * style_bottleneck_content \
      + (1 - content_blending_ratio) * style_bottleneck

  # Stylize the content image using the style bottleneck.
  stylized_image = run_style_transform(
      style_bottleneck_blended, preprocessed_content_image)

  print('🐌', end="")
  return stylized_image

def sigmoid(x):
  y = np.zeros(len(x))
  for i in range(len(x)):
    y[i] = 255 / (1 + math.exp(-x[i]))
  return y

# generate sigmoid.
# size: the width and height in pixels
# magnitude: the strength of the sigmoid - higher values = sharper transition.
#            must be >0, increases above 100 don't make much difference
# squeeze: on the outer sides of the sigmoid, have a dead zone. ex. a squeeze
#          of half the size means we'll vary the pixels by the sigmoid function
#          of the inner half of the tile only. this should probably be a multiple
#          of two or maybe everything will break??
# flip: if true, values change across the y-axis
def generate_sigmoid(size, magnitude, squeeze, flip=False):
  m = magnitude/10
  adj_size = size-squeeze
  sigmoid_ = sigmoid(np.concatenate((
      np.repeat(-m, squeeze/2),
      np.arange(-m, m, 4*m/adj_size),
      np.arange(m, -m, -4*m/adj_size),
      np.repeat(-m, squeeze/2)
  )))
  alpha = np.repeat(sigmoid_.reshape((len(sigmoid_), 1)), repeats=size, axis=1)
  if flip:
    alpha = np.swapaxes(alpha, 0, 1)
  res = Image.fromarray(np.uint8(alpha), 'L')
  return res

def sew(stylized_pieces, cfg):
  cols = cfg.cols
  rows = cfg.rows
  edge_size = cfg.edge_size
  wh = cfg.content_size

  im = Image.new('RGB', (
      cols*wh-2*edge_size*(cols-1),
      rows*wh-2*edge_size*(rows-1)
  ))

  for r in range(0, rows):
    for c in range(0, cols):
      squeezed = tf.squeeze(stylized_pieces[c+r*cols])
      imaged = tf.keras.preprocessing.image.array_to_img(squeezed)

      el = 0 if c == 0 else edge_size
      er = 0 if c == cols-1 else edge_size
      eu = 0 if r == 0 else edge_size
      ed = 0 if r == rows-1 else edge_size

      cropped = imaged.crop((el, eu, wh-er, wh-ed))

      im.paste(cropped, (
          c*wh - max(0, (2*c-1)*el),  # left
          r*wh - max(0, (2*r-1)*eu)  # up
      ))
  return im

def apply_row_joints(orig, joints, cfg):
  cols = cfg.cols
  rows = cfg.rows
  edge_size = cfg.edge_size
  wh = cfg.content_size
  magnitude = cfg.magnitude
  squeeze = cfg.squeeze

  updated_image = orig.copy()
  mask = generate_sigmoid(384, magnitude, squeeze, True)
  assert(rows*(cols-1) == len(joints))

  for r in range(0, rows):
    for c in range(0, cols-1):
      squeezed = tf.squeeze(joints[c+r*(cols-1)])
      imaged = tf.keras.preprocessing.image.array_to_img(squeezed)

      el = 0 if c == 0 else edge_size
      er = 0 if c == cols-1 else edge_size
      eu = 0 if r == 0 else edge_size
      ed = 0 if r == rows-1 else edge_size

      cropped = imaged.crop((el, eu, wh-er, wh-ed))

      # blackbox shows you the waveform of what gets converted
      # blackbox = Image.new('RGB', cropped.size, 0)
      updated_image.paste(cropped, (
          int((0.5+c)*wh - (0.5*edge_size + 1.5*edge_size*c)),  # left
          r*wh - max(0, (2*r-1)*eu)  # up
      ), mask.resize(cropped.size))
  return updated_image


def apply_column_joints(orig, joints, cfg):
  cols = cfg.cols
  rows = cfg.rows
  edge_size = cfg.edge_size
  wh = cfg.content_size
  squeeze = cfg.squeeze
  magnitude = cfg.magnitude

  updated_image = orig.copy()
  mask = generate_sigmoid(384, magnitude, squeeze, False)
  assert(((rows-1)*cols) == len(joints))

  for r in range(0, rows-1):
    for c in range(0, cols):
      squeezed = tf.squeeze(joints[c+r*cols])
      imaged = tf.keras.preprocessing.image.array_to_img(squeezed)

      el = 0 if c == 0 else edge_size
      er = 0 if c == cols-1 else edge_size
      eu = 0 if r == 0 else edge_size
      ed = 0 if r == rows-1 else edge_size

      cropped = imaged.crop((el, eu, wh-er, wh-ed))

      # blackbox shows you the waveform of what gets converted
      # blackbox = Image.new('RGB', cropped.size, 0)
      updated_image.paste(cropped, (
          c*wh - max(0, (2*c-1)*el),  # left
          int((0.5+r)*wh - (0.5*edge_size + 1.5*edge_size*r))  # up
      ), mask.resize(cropped.size))
  return updated_image

def get_nice_name(path):
  output = path

  try:
    # unix
    output = re.search(r'.*\\(.*).jpg', path).group(1)
  except AttributeError:
    try:
      # windows
      output = re.search(r'.*/(.*).jpg', path).group(1)
    except AttributeError:
        # no directory slashes
        output = re.search(r'(.*).jpg', path).group(1)

  return output

def get_output_filename(content_image_path, style_image_path, 
                        content_blending_ratio, cols, rows,
                        edge_size, use_tiled_style_image, use_fluid_blend, 
                        magnitude, squeeze,
                        extra_id="", **kwargs):
  content_blending_ratio = float(content_blending_ratio)
  output = "{0}+{1}-hd-fusion-{2}x{3}-blend{4}{5}{6}{7}{8}.jpg".format(
      get_nice_name(content_image_path),
      get_nice_name(style_image_path),
      cols, 
      rows,
      int(10*content_blending_ratio),
      "-edge{0}".format(edge_size) if edge_size > 0 else "",
      "-tiled" if use_tiled_style_image else "",
      "-fluid{0}-sq{1}".format(magnitude, squeeze) if use_fluid_blend else "",
      extra_id
  )
  return output

def run(
    content_image_path,
    style_image_path,
    output_directory,
    drive_base="",
    cols=1,
    rows=1,
    use_tiled_style_image=False,
    use_fluid_blend=True,
    edge_size=0,
    magnitude=30,
    squeeze=96,
    content_blending_ratio=0.5,
    content_blending_ratios=[],
    **kwargs
):
  config = dict(cols=cols, rows=rows, edge_size=edge_size, squeeze=squeeze,
                magnitude=magnitude, content_size=CONTENT_SIZE, style_size=STYLE_SIZE)
  config = SimpleNamespace(**config)
  print("content: {0}, style: {1}".format(content_image_path, style_image_path))
  print(config)

  if len(content_blending_ratios) == 0:
    content_blending_ratios = [content_blending_ratio]

  content_image = Image.open(drive_base+content_image_path)
  style_image = Image.open(drive_base+style_image_path)

  content_pieces = get_array_of_pieces(content_image, config)
  if use_tiled_style_image:
    style_pieces = get_array_of_pieces(style_image, config)
  else:
    style_pieces = list([style_image])

  if use_fluid_blend:
    (row_joints, col_joints) = get_intermediate_tiles(content_image, config)
    content_pieces.extend(row_joints)
    content_pieces.extend(col_joints)

  tf_content_pieces = list(
      map(lambda x: load_content_img(np.array(x)), content_pieces))
  preprocessed_content_pieces = list(
      map(lambda x: preprocessor(x, CONTENT_SIZE), content_pieces))
  preprocessed_style_pieces = list(
      map(lambda x: preprocessor(x, STYLE_SIZE), style_pieces))

  style_bottlenecks = list(
      map(lambda x: run_style_predict(x), preprocessed_style_pieces))
  style_bottleneck_contents = list(map(lambda x: run_style_predict(
      preprocess_image(x, STYLE_SIZE)), tf_content_pieces))

  print("Processing", len(preprocessed_content_pieces), "cells of content...")

  # loop for crankin out more blends
  for content_blending_ratio in content_blending_ratios:
    content_blending_ratio = float(content_blending_ratio)
    print("content blend:", content_blending_ratio)
    if use_tiled_style_image:
      stylized_pieces = list(map(lambda x, y, z: stylize(
          x, y, z, content_blending_ratio), preprocessed_content_pieces, style_bottlenecks, style_bottleneck_contents))
    else:
      stylized_pieces = list(map(lambda x, z: stylize(
          x, style_bottlenecks[0], z, content_blending_ratio), preprocessed_content_pieces, style_bottleneck_contents))

    image = sew(stylized_pieces, config)
    if use_fluid_blend:
      row_joints = stylized_pieces[(rows*cols):(rows*cols+rows*(cols-1))]
      column_joints = stylized_pieces[(rows*cols+rows*(cols-1)):]
      image = apply_column_joints(
          apply_row_joints(image, row_joints, config), column_joints, config)

    output_filename = "{0}/{1}".format(output_directory, get_output_filename(
        content_image_path=content_image_path,
        style_image_path=style_image_path,
        content_blending_ratio=content_blending_ratio,
        rows=rows,
        cols=cols,
        edge_size=edge_size,
        use_tiled_style_image=use_tiled_style_image,
        use_fluid_blend=use_fluid_blend,
        magnitude=magnitude,
        squeeze=squeeze
    ))
    image.save(output_filename, "JPEG")
    print("Saved to:", output_filename)

  # note that this is only the final image if you used multiple blending ratios
  return image
