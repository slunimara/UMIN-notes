import os, shutil

# The path to the folder
original_dir = "dogs-vs-cats/train"

# In the script folder create folder for train, validation and test
train_dir = os.path.join(os.curdir, 'train_subset')
validation_dir = os.path.join(os.curdir, 'validation_subset')
test_dir = os.path.join(os.curdir, 'test_subset')

num_for_training = 2000
num_for_validation = 1000
num_for_testing = 1000

params = zip([train_dir, validation_dir, test_dir],
           [num_for_training, num_for_validation, num_for_testing])

start = 0
for p in params:
    dir, num_of_images = p
    os.mkdir(dir)

    os.mkdir(os.path.join(dir, 'cats'))
    os.mkdir(os.path.join(dir, 'dogs'))
    
    # Because I have 2 classes. I`ll copy 1000 images for dogs and 1000 for cats to train folder.
    num_of_images = (int) (num_of_images / 2)
    for animal in ['cat', 'dog']:
        file_names = [f"{animal}.{i}.jpg" for i in range(start, start + num_of_images)]
        for file_name in file_names:
            src = os.path.join(original_dir, file_name)
            dst = os.path.join(dir, f"{animal}s", file_name)
            shutil.copyfile(src, dst)
    
    start += num_of_images

# Sanity check
for dir in [train_dir, validation_dir, test_dir]:
    for animal in ['cats', 'dogs']:
        print(f"Number of {animal} images in {dir}: {len(os.listdir(os.path.join(dir, animal)))}")