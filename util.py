from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch 
from scipy.io import loadmat
import os
from scipy.signal import decimate
import torchvision.models as models
from pathlib import Path

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

############################# Low-level Functions: Dataloaders

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
    

class NSD_ImageDataset(Dataset):
    def __init__(self, imgs_paths, transform):
        self.imgs_paths = np.array(imgs_paths)
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image at the given index
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path) # .convert('RGB')

        # Apply transformations if provided
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

def get_nsd_dataloader(transform, train_imgs_paths, test_imgs_paths, batch_size=128, num_workers=4):
    """Function to get the dataloader for the NSD dataset"""
    
    train_dataset = NSD_ImageDataset(transform=transform, paths=train_imgs_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    test_dataset = NSD_ImageDataset(transform=transform, paths=test_imgs_paths)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, test_dataloader


################################# Low-level Functions: Neurodata

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


def get_eeg(subject, path_to_eeg = None, group = None, downsample_factor = None, average_trials = False, time_range = None):
    """
    Dataloader for EEG data

    Args:
        subject (str): Subject ID (e.g. "01")
        path_to_eeg (str): Path to the preprocessed EEG data directory
        group (str): Group name (e.g. "training" or "test")
    Returns:
        dict_keys(['preprocessed_eeg_data', 'ch_names', 'times'])
    """

    if path_to_eeg is None:
        path_to_eeg = os.path.expanduser("~/Documents/BrainAlign_Data/eeg_preprocessed")
    
    eeg_subject = np.load(os.path.join(path_to_eeg, f"sub-{subject}/preprocessed_eeg_{group}.npy"), allow_pickle=True).item()

    #round time
    eeg_subject['times'] = eeg_subject['times'].round(2)

    if downsample_factor is not None:
        eeg_subject['preprocessed_eeg_data'] = decimate(eeg_subject['preprocessed_eeg_data'], downsample_factor, axis=-1, ftype='fir', zero_phase=True)
        eeg_subject['times'] = eeg_subject['times'][::downsample_factor]
    if average_trials:
        eeg_subject["preprocessed_eeg_data"] = eeg_subject["preprocessed_eeg_data"].mean(axis=1)

    if time_range is not None:
        time_mask = (eeg_subject['times'] >= time_range[0]) & (eeg_subject['times'] <= time_range[1])
        eeg_subject['times'] = eeg_subject['times'][time_mask]
        if average_trials:
            eeg_subject['preprocessed_eeg_data'] = eeg_subject['preprocessed_eeg_data'][:, :, time_mask]
        else:
            eeg_subject['preprocessed_eeg_data'] = eeg_subject['preprocessed_eeg_data'][:, :, :,time_mask]

    return eeg_subject


def get_fmri(subject, hemisphere, path_to_fmri = None):
    """
    Dataloader for fMRI data:
    Args:
        subject (str): Subject ID (e.g. "01")
        hemishere (str): Hemishere name (e.g. "lh" or "rh")
    Returns:
        dict_keys(['train_data', 'test_data', 'roi_masks'])
        roi_masks: dict_keys(['V1', 'V4', 'FFA'])
    
    """
    
    if path_to_fmri is None:
        path_to_fmri = os.path.expanduser("~/Documents/BrainAlign_Data/NSD_preprocessed")


    train_data = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/training_split/training_fmri/{hemisphere}_training_fmri.npy"))
    test_data = np.load(os.path.join(path_to_fmri, f"test_data/subj{subject}/test_split/test_fmri/{hemisphere}_test_fmri.npy"))

    # load two mask files
    mask_early = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.prf-visualrois_challenge_space.npy"))
    mask_ffa = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.floc-faces_challenge_space.npy"))
    
    # load two mapping files
    mapping_early = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/mapping_prf-visualrois.npy"),
                            allow_pickle=True).item()
    mapping_ffa = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/mapping_floc-faces.npy"),
                        allow_pickle=True).item()
    
    # create masks
    # create masks for brain areas 
    keys = [k for k, v in mapping_early.items() if v in ['V1v', 'V1d']]
    V1_mask = np.isin(mask_early, keys)

    keys = [k for k, v in mapping_early.items() if v in ['hV4']]
    V4_mask = np.isin(mask_early, keys)

    keys = [k for k, v in mapping_ffa.items() if v in ['FFA-1', 'FFA-2']]
    FFA_mask = np.isin(mask_ffa, keys)

    # create dict for masks
    masks = {
        'V1': V1_mask,
        'V4': V4_mask,
        'FFA': FFA_mask
    }

    output = {
        'train_data': train_data,
        'test_data': test_data,
        'roi_masks': masks
    }

    return output


def get_fmri_2 (subject, hemisphere, path_to_fmri = None):
    """
    Dataloader for fMRI data:
    Args:
        subject (str): Subject ID (e.g. "01")
        hemishere (str): Hemishere name (e.g. "lh" or "rh")
    Returns:
        dict_keys(['train_data', 'test_data', 'roi_masks'])
        roi_masks: dict_keys(['V1', 'V4', 'FFA'])
    
    """
    
    if path_to_fmri is None:
        path_to_fmri = os.path.expanduser("~/Documents/BrainAlign_Data/NSD_preprocessed")

    # load training and test data
    train_data = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/training_split/training_fmri/{hemisphere}_training_fmri.npy"))
    test_data = np.load(os.path.join(path_to_fmri, f"test_data/subj{subject}/test_split/test_fmri/{hemisphere}_test_fmri.npy"))

    # load two mask files
    # challenge space (for indexing of data)
    mask_early = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.prf-visualrois_challenge_space.npy"))
    mask_ffa = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.floc-faces_challenge_space.npy"))
    # fsaverage space (for visualization)
    mask_early_fsl = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.prf-visualrois_fsaverage_space.npy"))
    mask_ffa_fsl = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/{hemisphere}.floc-faces_fsaverage_space.npy"))
    
    # load two mapping files
    mapping_early = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/mapping_prf-visualrois.npy"),
                            allow_pickle=True).item()
    mapping_ffa = np.load(os.path.join(path_to_fmri, f"train_data/subj{subject}/roi_masks/mapping_floc-faces.npy"),
                        allow_pickle=True).item()
    

    # create masks
    V1_keys = [k for k, v in mapping_early.items() if v in ['V1v', 'V1d']]
    V1_mask = np.isin(mask_early, V1_keys)
    V1_mask_fsl = np.isin(mask_early_fsl, V1_keys)

    V4_keys = [k for k, v in mapping_early.items() if v in ['hV4']]
    V4_mask = np.isin(mask_early, V4_keys)
    V4_mask_fsl = np.isin(mask_early_fsl, V4_keys)

    FFA_keys = [k for k, v in mapping_ffa.items() if v in ['FFA-1', 'FFA-2']]
    FFA_mask = np.isin(mask_ffa, FFA_keys)
    FFA_mask_fsl = np.isin(mask_ffa_fsl, FFA_keys)
    
    # combine the masks
    combined_mask = np.zeros_like(FFA_mask, dtype=int)  # Initialize with zeros
    fsl_mask = np.zeros_like(FFA_mask_fsl, dtype=int)  # Initialize with zeros

    mask_mapping = {
        1: "V1",
        2: "V4",
        3: "FFA"
    }

    combined_mask[V1_mask] = 1 ; fsl_mask[V1_mask_fsl] = 1
    combined_mask[V4_mask] = 2 ; fsl_mask[V4_mask_fsl] = 2
    combined_mask[FFA_mask] = 3; fsl_mask[FFA_mask_fsl] = 3

    output = {
        'train_data': train_data,
        'test_data': test_data,
        'mask_challenge': combined_mask,
        'mask_fsl': fsl_mask,
        'mask_mapping': mask_mapping,

    }

    return output


def get_fmri_concat_hemispheres(lh_dat, rh_dat):

    """
    Function to concatenate hemisphere data while preserving indices with respect to fsl average space
    
    """
    
    train_data = np.hstack([lh_dat["train_data"], rh_dat["train_data"]])
    test_data = np.hstack([lh_dat["test_data"], rh_dat["test_data"]])

    mask_challenge = {
        'roi' : np.concatenate([lh_dat["mask_challenge"], rh_dat["mask_challenge"]]),
        'hemisphere' : np.concatenate([np.repeat('lh', lh_dat["mask_challenge"].shape), np.repeat('rh', rh_dat["mask_challenge"].shape)]),
    }

    mask_fsl = {
        'roi' : np.concatenate([lh_dat["mask_fsl"], rh_dat["mask_fsl"]]), 
        'hemisphere' : np.concatenate([np.repeat('lh', lh_dat["mask_fsl"].shape), np.repeat('rh', rh_dat["mask_fsl"].shape)])
    }


    output = {
        'train_data': train_data,
        'test_data': test_data,
        'mask_challenge': mask_challenge,
        'mask_fsl': mask_fsl,
        'mask_mapping': rh_dat['mask_mapping'],

    }

    return output







def get_model(model_name, seed):
    if model_name == 'alexnet':
        #model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        model = models.alexnet()

        # Step 2: Load the full dict
        path = os.path.expanduser(f'~/Documents/pretrained_DNNs/alexnet_model_89_s{seed}.pth')
        checkpoint = torch.load(path, weights_only=False)

        # Step 3: Extract the model weights
        state_dict = checkpoint['model']

        # Step 4: Load into model
        model.load_state_dict(state_dict)

        # Step 5: Evaluation mode (optional)
        model.eval()

    elif model_name == 'resnet50':

        model = models.resnet50()

        path = os.path.expanduser(f'~/Documents/pretrained_DNNs/resnet50_model_89_s{seed}.pth')
        checkpoint = torch.load(path, weights_only=False)

        state_dict = checkpoint['model']

        model.load_state_dict(state_dict)
        model.eval()

    return model


############################# High-level Functions


def get_neurodata(dataset_name, subjects, ephys_normalized=True):
    if dataset_name == "tvsd":
        return get_tvsd(subjects, normalized=ephys_normalized)
    elif dataset_name == "eeg":
        return get_eeg(subjects)
    elif dataset_name == "fmri":
        return get_fmri(subjects)



def get_dataloader(dataset_name, batch_size=128, num_workers=4, subject = None):
    """Function to get the dataloader for the specified dataset
    
    Args:
        dataset_name (str): The name of the dataset to get the dataloader for
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader
        
    Returns:
        train_dataloader, test_dataloader: The dataloaders for the training and testing sets
    """

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

        train_imgs_paths = get_image_paths_eeg(group_name = 'train')
        test_imgs_paths = get_image_paths_eeg(group_name = 'test')

        for i in range(len(train_imgs_paths)):
            train_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(train_imgs_paths[i]))
        
        for i in range(len(test_imgs_paths)):
            test_imgs_paths[i] = os.path.join(THINGS_PATH, os.path.normpath(test_imgs_paths[i]))

        train_dataloader, test_dataloader = get_things_dataloader(transform,THINGS_PATH, train_imgs_paths, test_imgs_paths, batch_size=batch_size, num_workers=num_workers)

        return train_dataloader, test_dataloader
    
    elif dataset_name == 'fmri':
        # note that here we have to load per subject
        if subject is None:
            raise ValueError("Subject ID must be provided for fMRI data.")
        
        path_to_fmri = os.path.expanduser("~/Documents/BrainAlign_Data/NSD_preprocessed")
        train_img_dir = os.path.join(path_to_fmri, f"train_data/subj{subject}/training_split/training_images")
        test_img_dir = os.path.join(path_to_fmri, f"test_data/subj{subject}/test_split/test_images")

        #train_img_list = os.listdir(train_img_dir)
        #train_img_list.sort()
        #test_img_list = os.listdir(test_img_dir)
        #test_img_list.sort()

        train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
        test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

        train_dataloader = DataLoader(
            NSD_ImageDataset(train_imgs_paths, transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle =False
        )
        test_dataloader = DataLoader(
            NSD_ImageDataset(test_imgs_paths, transform),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle = False
        )
        
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
                all_features[key].append(activation.detach().cpu())  # Detach and move to CPU for storage
    # Concatenate all batch features for each layer
    all_features = {key: torch.cat(features, dim=0) for key, features in all_features.items()}
    
    return all_features

