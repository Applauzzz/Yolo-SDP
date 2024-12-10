import os
import shutil
import random

root_dir = '/mnt/nvme_storage/database/Nonmine_site'
activate_dir = os.path.join(root_dir, 'activate_site')
abandon_dir = os.path.join(root_dir, 'abandon_site')
train_dir = os.path.join(root_dir, 'train-v2')
test_dir = os.path.join(root_dir, 'test-v2')
val_dir = os.path.join(root_dir, 'val-v2')

cat_list = ['Mining-Building', 'Mining-Costean_Trench', 'Mining-Dam','Mining-Rubbish Dump', 'Mining-Shaft']

num_images = 1000

for category in os.listdir(activate_dir):
    if category not in cat_list:
        continue
    category_path = os.path.join(activate_dir, category)
    
    if os.path.isdir(category_path):
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        
        selected_images = images
        total_images = len(selected_images)
        num_train = int(0.8*total_images)
        num_test = int(0.2 * num_train)
        num_val = num_test

        train_images = selected_images[:num_train]
        test_images = selected_images[num_train:num_test+num_train]
        val_images = test_images
        print(len(train_images), len(test_images), len(val_images))
        for subset, image_list in zip(['train-v2', 'test-v2', 'val-v2'], [train_images, test_images, val_images]):
            tt = 'activate_' + category
            subset_dir = os.path.join(root_dir, subset, tt)
            os.makedirs(subset_dir, exist_ok=True)
            
            for img in image_list:
                img_path = os.path.join(category_path, img)
                dest_path = os.path.join(subset_dir, img)
                shutil.copy(img_path, dest_path)

for category in os.listdir(abandon_dir):
    if category not in cat_list:
        continue
    category_path = os.path.join(abandon_dir, category)
    
    if os.path.isdir(category_path): 
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        
        selected_images = images
        total_images = len(selected_images)
        num_train = int(0.8*total_images)
        num_test = int(0.2 * num_train)
        num_val = num_test  

        train_images = selected_images[:num_train]
        test_images = selected_images[num_train:num_test+num_train]
        val_images = test_images
        for subset, image_list in zip(['train-v2', 'test-v2', 'val-v2'], [train_images, test_images, val_images]):
            tt = 'abandon_' + category
            subset_dir = os.path.join(root_dir, subset, tt)
            os.makedirs(subset_dir, exist_ok=True)
            
            for img in image_list:
                img_path = os.path.join(category_path, img)
                dest_path = os.path.join(subset_dir, img)
                shutil.copy(img_path, dest_path)

category_path = "/mnt/nvme_storage/database/Nonmine_site/Mining-Nonemine"
    
if os.path.isdir(category_path): 
    images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
    print(len(images))
    selected_images = images
    total_images = len(selected_images)
    num_train = int(0.8*total_images)
    num_test = int(0.2 * num_train)
    num_val = num_test 

    train_images = selected_images[:num_train]
    test_images = selected_images[num_train:num_test+num_train]
    val_images = test_images

    for subset, image_list in zip(['train-v2', 'test-v2', 'val-v2'], [train_images, test_images, val_images]):
        tt = "Nonmine"
        subset_dir = os.path.join(root_dir, subset, tt)
        os.makedirs(subset_dir, exist_ok=True)
        
        for img in image_list:
            img_path = os.path.join(category_path, img)
            dest_path = os.path.join(subset_dir, img)
            shutil.copy(img_path, dest_path)


