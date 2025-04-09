from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch 
from scipy.io import loadmat
import os

#################### Get Image Paths for THINGS dataset ################


def get_image_paths(file_path, group_name, num_images=None):
    with h5py.File(file_path, "r") as file:
        group = file[group_name]
        things_path_refs = group['things_path']
        image_paths = []

        # Determine the number of images to process
        total_images = len(things_path_refs) if num_images is None else min(num_images, len(things_path_refs))

        for ref in things_path_refs[:total_images]:
            ref_obj = file[ref.item()]  
            path_str = ''.join(chr(c) for c in ref_obj[:].flatten())  
            path_str = path_str.replace("\\", "/")  # Ensure compatibility across OS

            # Prepend correct image directory
            full_path = os.path.join(os.path.expanduser("~/Documents/BrainAlign_Data/things_images/"), path_str)
            image_paths.append(full_path)
        
    return image_paths



def get_image_paths_eeg(group_name = None):

    """
    get image paths for training or test set using image metadate .npy file
    group_name = "test" or "train"
    """

    # Load the metadata file
    metadata_path = os.path.expanduser("~/Documents/BrainAlign_Data/eeg_image_metadata.npy")
    metadata = np.load(metadata_path, allow_pickle=True).item()

    # Get the image paths for the specified group
    path_strings = [f"{category.split('_', 1)[1]}/{image}" for category, image in zip(metadata[f"{group_name}_img_concepts"], metadata[f"{group_name}_img_files"])]

    image_paths = [os.path.join(os.path.expanduser("~/Documents/BrainAlign_Data/things_images/"), path_string) for path_string in path_strings]

    return image_paths

############################# Low-level Functions: Dataset

class THINGS(Dataset):
    def __init__(self, root, paths, transform=None, device='cuda'):
        self.paths = paths
        self.transform = transform
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(os.path.join(path))

        if self.transform:
            img = self.transform(img)
        return img, 0., idx 

def get_things_dataloader(transform, THINGS_PATH,train_imgs_paths, test_imgs_paths, batch_size=128, num_workers=4):
    """Function to get the dataloader for the THINGS dataset"""
    
    train_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=train_imgs_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    test_dataset = THINGS(root=THINGS_PATH, transform=transform, paths=test_imgs_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, test_dataloader


def get_tvsd(subject_file_path, normalized=True, device="cuda", group_name = "train_MUA", monkey = None):
    """Dataloader for neurodata from TVSD dataset, this function will return the data for a single subject separated into V1, V4, IT"""

    # Load the data from the h5 file
    with h5py.File(subject_file_path, "r") as file:
        data_dict = {key: torch.tensor(np.array(file[key]), dtype=torch.float32, device=device) for key in file.keys()}

    train_mua = data_dict.get(group_name)
    
    if monkey == "n":
        V1 = train_mua[:, :512]  #1:512 =  V1
        V4 = train_mua[:, 512:768] #513:768 = V4
        IT = train_mua[:, 768:1024] #769:1024 = IT
    elif monkey == "f":
        V1 = train_mua[:, :512] #1:512 =  V1
        V4 = train_mua[:, 833:1024] #833:1024 = V4
        IT = train_mua[:, 513:832] #513:832 = IT

    
    return V1, V4, IT  # Return the data for V1, V4, IT

# get custom object from tvsd file (e.g. SNR)

def get_tvsd_custom(subject_file_path, device="cuda", group_name = ""):

    """
    get custom object from tvsd file (e.g. SNR)
    """
    # Load the data from the h5 file
    with h5py.File(subject_file_path, "r") as file:
        data_dict = {key: torch.tensor(np.array(file[key]), dtype=torch.float32, device=device) for key in file.keys()}

    object = data_dict.get(group_name)

    return(object)


def get_eeg(subject):
    """Dataloader for EEG data"""
    pass


def get_fmri(subject):
    pass




def get_model(model_name):
    if model_name == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        #Add more later
    return model


############################# High-level Functions


def get_neurodata(dataset_name, subjects, ephys_normalized=True):
    if dataset_name == "tvsd":
        return get_tvsd(subjects, normalized=ephys_normalized)
    elif dataset_name == "eeg":
        return get_eeg(subjects)
    elif dataset_name == "fmri":
        return get_fmri(subjects)



def get_dataloader(dataset_name, batch_size=128, num_workers=4):
    """Function to get the dataloader for the specified dataset
    
    Args:
        dataset_name (str): The name of the dataset to get the dataloader for
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader
        
    Returns:
        train_dataloader, test_dataloader: The dataloaders for the training and testing sets
    """
    from sklearn.model_selection import train_test_split

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check which dataset and respond accordingly
    if dataset_name == 'tvsd':
        
        # define directory where THINGS images are stored
        THINGS_PATH = os.path.expanduser("~/Documents/BrainAlign_Data/things_images/")

        # define path to mat files where image paths are stored
        file_path = os.path.expanduser("~/Documents/BrainAlign_Data/things_imgs.mat")
        train_imgs_paths = get_image_paths(file_path, 'train_imgs')
        test_imgs_paths = get_image_paths(file_path, 'test_imgs')

        for i in range(len(train_imgs_paths)):
            train_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(train_imgs_paths[i]))
        
        for i in range(len(test_imgs_paths)):
            test_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(test_imgs_paths[i]))

        # The below part should then go into the get_things_dataloader function which you then only call here
        train_dataloader, test_dataloader = get_things_dataloader(transform,THINGS_PATH, train_imgs_paths, test_imgs_paths, batch_size=batch_size, num_workers=num_workers)

        return train_dataloader, test_dataloader

    elif dataset_name == 'eeg':

        THINGS_PATH = os.path.expanduser("~/Documents/BrainAlign_Data/things_images/")

        train_imgs_paths = get_image_paths_eeg(group_name = "train")
        test_imgs_paths = get_image_paths_eeg(group_name = "test")

        for i in range(len(train_imgs_paths)):
            train_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(train_imgs_paths[i]))
        
        for i in range(len(test_imgs_paths)):
            test_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(test_imgs_paths[i]))

        train_dataloader, test_dataloader = get_things_dataloader(transform,THINGS_PATH, train_imgs_paths, test_imgs_paths, batch_size=batch_size, num_workers=num_workers)

        return train_dataloader, test_dataloader



    elif dataset_name == 'NSD':
        ... # do NSD dataloader preparation

        return train_dataloader, test_dataloader
    

    

################################## Feature Extraction functions
import torch
import tqdm
from torchvision.models.feature_extraction import create_feature_extractor

def extract_features(model, dataloader, device, return_nodes=None, unflatten=True):
    """
    Extracts features from specific layers of a pre-trained model.

    Args:
        model (torch.nn.Module): Pretrained model for feature extraction.
        dataloader (DataLoader): DataLoader to fetch images in batches.
        device (torch.device): Device to run inference on (CPU/GPU).
        return_nodes (dict, optional): Dictionary mapping layer names to output names.
        unflatten (bool, optional): If True, flattens the output features.
    
    Returns:
        dict: Dictionary of extracted features from specified layers.
    """
    # Apply return_nodes if specified
    if return_nodes:
        model = create_feature_extractor(model, return_nodes=return_nodes)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval() 
    
    # Initialize dictionary to store features for each layer with the key as the layer name
    all_features = {key: [] for key in return_nodes.values()}  


    # Iterate over the dataloader to extract features
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader, total=len(dataloader)):
            imgs, _, _ = item  # Unpack all three returned values
            imgs = imgs.to(device)
            
            batch_activations = model(imgs) 
            for key, activation in batch_activations.items():
                if unflatten:
                    activation = torch.flatten(activation, start_dim=1)  # Flatten while keeping batch dim
                all_features[key].append(activation.cpu())
    # Concatenate all batch features for each layer
    all_features = {key: torch.cat(features, dim=0) for key, features in all_features.items()}
    
    return all_features

