import os, random
import time
import style_transfer_utils as sxu

content_folder = "F:\\Google Drive\\images\\prettyearth\\"
style_folder = "F:\\Google Drive\\images\\mtg\\"
output_directory = "F:\\Google Drive\\images\\generated-20210321\\"

def does_file_already_exist(filename):
  return os.path.isfile(output_directory + filename)

def dream():
  content_image_path = content_folder + random.choice(os.listdir(content_folder))
  style_image_path = style_folder + random.choice(os.listdir(style_folder))
  # print(content_image_path)
  # print(style_image_path)
  params = dict(
    drive_base="",
    content_image_path=content_image_path,
    style_image_path=style_image_path,
    rows=1,
    cols=1,
    use_tiled_style_image=False,
    use_fluid_blend=False,
    edge_size=8,
    magnitude=2,
    content_blending_ratio=0.5,
    output_directory=output_directory
  )

  if does_file_already_exist(sxu.get_output_filename(**params)):
    return

  sxu.run(**params)
  return

while True:
  dream()
  time.sleep(1)
