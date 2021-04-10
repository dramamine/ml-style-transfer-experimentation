# waifu2x-ncnn-vulkan.exe -i 7008-Voice-of-All-Planeshft-MtG-Art-hd-fusion-3x2-blend1-fluid30-sq96.jpg -o 7008-006.png -n 3
import sys, os, re, subprocess

source_dir = sys.argv[1]
print(f'Source directory is {source_dir}')


def process(input_file, output_file):
  if os.path.isfile(source_dir + output_file):
    print("That file already exists: ", output_file)
    return
  subprocess.run(["waifu2x-ncnn-vulkan.exe", "-i", input_file, "-o", output_file, "-n", "3"], cwd=source_dir)
  print(output_file)
  pass

for file in os.listdir(source_dir):
  if file.endswith('.jpg'):
    prettyearth_digits_re = re.search('^(\d+).*$', file)
    if not prettyearth_digits_re:
      continue
    prettyearth_digits = prettyearth_digits_re.group(1)

    blend_digits_re = re.search('^.*blend(\d+)-.*$', file)
    if not blend_digits_re:
      process(file, "{0}-000.png".format(prettyearth_digits))
      continue
    blend_digits = int( (9 - int(blend_digits_re.group(1))) / 2) + 1
    process(file, "{0}-00{1}.png".format(prettyearth_digits, blend_digits))
