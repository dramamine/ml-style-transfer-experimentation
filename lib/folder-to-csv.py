import sys, os, re, subprocess

source_dir = sys.argv[1]
content_dir = sys.argv[2]
style_dir = sys.argv[3]
print(f'Source directory is {source_dir}')
print(f'Content directory is {content_dir}')
print(f'Style directory is {style_dir}')

sampler = [
    '3,2,,,,,0.3', # low correction
    '3,2,,,,30,288,0.3', # mid correction
    '3,2,,,48,336,0.3', # high correction
]

all_options = [
    '3,2,,,,,"[0.9,0.7,0.5,0.3,0.1]"',  # low correction
    '3,2,,,,30,288,"[0.9,0.7,0.5,0.3,0.1]"',  # mid correction
    '3,2,,,48,336,"[0.9,0.7,0.5,0.3,0.1]"',  # high correction
]

def image_to_path(dr, img):
  return "{0}\{1}.jpg".format(dr, img)


print("content_image_path, style_image_path, cols, rows, use_tiled_style_image, use_fluid_blend, edge_size, magnitude, content_blending_ratio")

print("# samples")
for file in os.listdir(source_dir):
  if file.endswith('.jpg'):
    filenames_re = re.search('^(.*)-hd-fusion.*$', file)
    if not filenames_re:
      continue
    
    images = filenames_re.group(1)
    [content, style] = images.split('+')
    starter = "{0},{1},".format(image_to_path(content_dir, content), image_to_path(style_dir, style))

    for sample in sampler:
      print(starter + sample)

# another loop lol.
for option in all_options:
  print("# correction levels")
  for file in os.listdir(source_dir):
    if file.endswith('.jpg'):
      filenames_re = re.search('^(.*)-hd-fusion.*$', file)
      if not filenames_re:
        continue

      images = filenames_re.group(1)
      [content, style] = images.split('+')
      starter = "{0},{1},".format(image_to_path(
          content_dir, content), image_to_path(style_dir, style))
      print(starter + option)

