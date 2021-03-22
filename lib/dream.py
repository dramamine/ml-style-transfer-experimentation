import os, random
import time
import style_transfer_utils as sxu

content_folder = "E:\\images\\prettyearth"
style_folder = "E:\\images\\mtg"
output_directory = "E:\\git\\style-transfer-hd"

def does_file_already_exist(params):
  return os.path.isfile(output_directory + "\\" + filename)

def dream():
  content_image_path = random.choice(os.listdir(content_folder))
  style_image_path = random.choice(os.listdir(style_folder))
  # print(content_image_path)
  # print(style_image_path)
  params = dict(
    drive_base="",
    content_image_path=content_image_path,
    style_image_path=style_image_path,
    rows=1,
    cols=1,
    use_tiled_style_image=False,
    use_fluid_blend=True,
    edge_size=8,
    magnitude=2,
    content_blending_ratio=0.5
  )

  if does_file_already_exist(sxu.get_output_filename(**params)):
    return

  sxu.run(**params)
  return

while True:
  dream()
  time.sleep(5)
