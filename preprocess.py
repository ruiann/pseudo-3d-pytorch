import subprocess
import os

height = 160
width = 160
frame = 4

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


data_dir = '/Users/inverse/Downloads/UCF-101'
target_dir = 'data'
mkdir(target_dir)

dir_list = os.listdir(data_dir)
for type in dir_list:
    dir_path = '{}/{}'.format(data_dir, type)
    if os.path.isdir(dir_path):
        target_type_path = '{}/{}'.format(target_dir, type)
        mkdir(target_type_path)
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            file_path = '{}/{}'.format(dir_path, file_name)
            image_path = '{}/{}'.format(target_type_path, file_name).replace('.avi', '')
            mkdir(image_path)
            shell = 'ffmpeg -i {0} -r {4} -s {2}x{3} -f image2 {1}/%03d.jpg'.format(file_path, image_path, width, height, frame)
            subprocess.call(shell, cwd=os.getcwd(), shell=True)