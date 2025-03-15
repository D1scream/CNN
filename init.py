import shutil
import random
import os

def reset_directory(dir_path):
    if len(os.listdir(dir_path)) == 0 :
        print(dir_path + " is already empty")
        return

    print("resetting "+ dir_path)
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(dir_path + " is now empty")

def copy_clean(src = '', dest ='', ignore = []):
    print("Copying file from " + src + " to " + dest)
    i = 0
    j = 0
    for filename in os.listdir(src):
        i += 1
        if filename not in ignore:
            shutil.copy(src + '/' + filename, dest + '/' + filename)
            j+=1
        if i % 1000 == 0:
            print(i, end=" ")
        if i % 10000 == 0:
            print()

    print(f"Copied {j} files. Copying complete.")

def split_data_to_dir(class_str = '', src_path = '', dest_path = '', samples = {}):
    src_path = src_path + '/' + class_str
    dest_path = dest_path + '/' + class_str
    print("Sending test samples to " + dest_path)
    i = 0
    for filename in samples[class_str]:
        shutil.copy(src_path + '/' + filename, dest_path + '/' + class_str + "_" + filename)
        i+=1
        if i % 250 == 0:
            print(i, end=" ")
        if i % 5000 == 0:
            print()

    print(f"Sent {i} test samples for {class_str}.")


def train_validation_split(data_path='', validation_split=0.15):
    data_files = os.listdir(data_path)
    data_size = len(data_files)
    validation_size = int(validation_split * data_size)
    
    validation_sample = random.sample(data_files, validation_size)
    train_sample = list(set(data_files) - set(validation_sample))
    
    print(f'Train size: {len(train_sample)}')
    print(f'Validation size: {len(validation_sample)}')
    
    return train_sample, validation_sample

def setup_directories():
    directories = [
        'working/dataset', 
        'working/train/cats', 
        'working/train/dogs',
        'working/test/cats', 
        'working/test/dogs', 
        'working/validation/cats', 
        'working/validation/dogs'
    ]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
def preprocess_data():
    original_dataset_path = 'OriginalDataset/train'
    clean_dataset_path = 'working/dataset'
    test_path = 'working/test'
    train_path = 'working/train'
    validation_path = 'working/validation'

    setup_directories()

    reset_directory(clean_dataset_path + '/cats')
    copy_clean(src=original_dataset_path + '/cats', dest=clean_dataset_path + '/cats')
    reset_directory(clean_dataset_path + '/dogs')
    copy_clean(src=original_dataset_path + '/dogs', dest=clean_dataset_path + '/dogs')

    cat_train_sample, cat_validation_sample = train_validation_split(data_path=clean_dataset_path + '/cats')
    dog_train_sample, dog_validation_sample = train_validation_split(data_path=clean_dataset_path + '/dogs')

    cat_test_sample = os.listdir(clean_dataset_path + '/cats')
    dog_test_sample = os.listdir(clean_dataset_path + '/dogs')

    test_samples = {'Cats': cat_test_sample, 'Dogs': dog_test_sample}
    validation_samples = {'Cats': cat_validation_sample, 'Dogs': dog_validation_sample}
    train_samples = {'Cats': cat_train_sample, 'Dogs': dog_train_sample}

    reset_directory(test_path + "/cats")
    split_data_to_dir("Cats", clean_dataset_path, test_path, test_samples)
    
    reset_directory(train_path + "/cats")
    split_data_to_dir("Cats", clean_dataset_path, train_path, train_samples)
    
    reset_directory(validation_path + "/cats")
    split_data_to_dir("Cats", clean_dataset_path, validation_path, validation_samples)


    reset_directory(test_path + "/dogs")
    split_data_to_dir("Dogs", clean_dataset_path, test_path, test_samples)
    
    reset_directory(train_path + "/dogs")
    split_data_to_dir("Dogs", clean_dataset_path, train_path, train_samples)
    
    reset_directory(validation_path + "/dogs")
    split_data_to_dir("Dogs", clean_dataset_path, validation_path, validation_samples)
