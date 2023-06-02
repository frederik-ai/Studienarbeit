import os
import shutil
import subprocess
import uuid

# Where all your pictograms are stored
all_pictograms_path = '../data/Pictograms Resized/'
# Emtpy folder where every single pictogram will be stores
single_pictogram_path = '../data/Single Pictogram/'
# Destination path
generated_images_path = 'generator_test/unet'

pictograms = os.listdir(all_pictograms_path)

for pictogram in pictograms:
  # Move pictogram
  current_pictogram_path = os.path.join(all_pictograms_path, pictogram)
  target_pictogram_path = os.path.join(single_pictogram_path, pictogram)
  shutil.copy(current_pictogram_path, target_pictogram_path)
  
  # Generate images
  cmd = ['python3', 'generate.py']
  subprocess.Popen(cmd).wait()
  
  # Move generated images to new folder
  unet_path_list = os.listdir(generated_images_path)
  class_id = str(uuid.uuid4())
  os.mkdir(os.path.join(generated_images_path, class_id))
      
  for element in unet_path_list:
    if os.path.isfile(os.path.join(generated_images_path, element)):
      shutil.move(os.path.join(generated_images_path, element), os.path.join(generated_images_path, class_id, element))
  
  # Remove old pictogram
  os.remove(target_pictogram_path)