import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    drive_path: str,  # Root path where Google Drive is mounted
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a root path where Google Drive is mounted, training directory, and
    testing directory path, and turns them into PyTorch Datasets and then into
    PyTorch DataLoaders.

    Args:
      drive_path: Root path where Google Drive is mounted in Colab.
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      transform: torchvision transforms to perform on training and testing data.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for the number of workers per DataLoader.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, test_dataloader, class_names = \
          create_dataloaders(drive_path='/content/drive',
                             train_dir='MyDrive/your_dataset_folder/train',
                             test_dir='MyDrive/your_dataset_folder/test',
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
    """
    # Construct full paths
    full_train_dir = os.path.join(drive_path, train_dir)
    full_test_dir = os.path.join(drive_path, test_dir)

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(full_train_dir, transform=transform)
    test_data = datasets.ImageFolder(full_test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
