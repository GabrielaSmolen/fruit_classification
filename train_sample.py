import os
import shutil
import os

src = 'data/fruits-360_dataset/fruits-360/Test'

os.makedirs('data/Test_small', exist_ok=True)

dest = 'data/Test_small'

for i in os.listdir(src):
    images = os.listdir(src + '/' + i)[:2]
    os.makedirs(dest + '/' + i, exist_ok=True)
    for img in images:
        shutil.copy(os.path.join(src, i, img), os.path.join(dest, i, img))
