# import marten_style_transfer_hd_fluid_edges
import csv
import os
import datetime
import style_transfer_utils
import hashlib
from ast import literal_eval


def does_file_already_exist(params):
  # @TODO call file-naming-function
  filename = 'process.py'
  return False
  #return os.path.isfile(filename)

def get_nums(s):
  valids = []
  for char in s:
    if char.isdigit() and len(valids) < 3:
      valids.append(char)
  return ''.join(valids)

def generate_stupid_hash(x, y):
  hexy = hashlib.blake2b("{0}x{1}".format(x, y).encode()).hexdigest()
  hex_string = hexy[10:18]
  nums = get_nums(hexy[20:30])
  return "{0}-{1}".format(nums, hex_string)

def add_defaults(data):
  if not data['cols']:
    data['cols'] = 1
  if not data['rows']:
    data['rows'] = 1 
  data['use_tiled_style_image'] = data['use_tiled_style_image'] == "True"
  data['use_fluid_blend'] = not data['use_fluid_blend'] == "False"
  if not data['edge_size']:
    data['edge_size'] = 8 
  if not data['magnitude']:
    data['magnitude'] = 2
  # if not data['content_blending_ratio']:
  #   data['content_blending_ratio'] = 0.5
  # else:
  #   data['content_blending_ratio'] = float(data['content_blending_ratio'])

  # do arrays
  data['content_image_paths'] = as_array(data['content_image_path'])
  data['style_image_paths'] = as_array(data['style_image_path'])
  data['content_blending_ratios'] = as_array(data['content_blending_ratio'])
  return data

def as_array(msg):
  if isinstance(msg, str) and '[' in msg:
    print("going to literal eval this:", msg)
    return literal_eval(msg)
  return [msg]


with open('jobs.csv') as csvfile:
  reader = csv.DictReader(csvfile, delimiter=",")
  for row in reader:
    row = add_defaults(row)

    print(row)

    for content_image_path in row['content_image_paths']:
      for style_image_path in row['style_image_paths']:
        for content_blending_ratio in row['content_blending_ratios']:
          
          row['content_image_path'] = content_image_path
          row['style_image_path'] = style_image_path
          row['content_blending_ratio'] = content_blending_ratio

          if does_file_already_exist(row):
            continue
          print(datetime.datetime.now())
          # marten_style_transfer_hd_fluid_edges.run(**row, drive_base="G:\\Google Drive\\images\\")
          
          name = style_transfer_utils.get_output_filename(**row)
          # gsh = generate_stupid_hash( row['content_image_path'], row['style_image_path'] )
          print(name)
          # print(gsh)
    
print("queue complete.")
