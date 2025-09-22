import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

# It should contain subfolders for each disease class.
TRAIN_DATA_PATH = 'downloads/train' 


# We just need to load the images to get the folder structure.
transform = transforms.Compose([transforms.ToTensor()])

print(f"Looking for class folders in: {TRAIN_DATA_PATH}")

if not os.path.exists(TRAIN_DATA_PATH):
    print(f"Error: The directory '{TRAIN_DATA_PATH}' was not found.")
    print("Please check the TRAIN_DATA_PATH variable in the script.")
else:
    try:
        # Load the dataset object 
        train_data = ImageFolder(TRAIN_DATA_PATH, transform=transform)

        class_names = train_data.classes

        if not class_names:
            print(f"No class subdirectories found in '{TRAIN_DATA_PATH}'.")
            print("Please ensure your training data is organized into subfolders.")
        else:
            with open('rice_leaf_classes.txt', 'w') as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")

            print("\nSuccessfully generated 'rice_leaf_classes.txt'")
            print("Found the following classes:")
            for i, name in enumerate(class_names):
                print(f"  {i}: {name}")

    except Exception as e:
        print(f"An error occurred: {e}")