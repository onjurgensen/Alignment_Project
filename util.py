from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
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

def get_tvsd(subject, device="cuda", group_name = "test_MUA", snr_cutoff = 2):

    """
    Load the TVSD data from the specified subject file.
    Args:
        subject (str): Subject identifier ("F" or "N").
        device (str): Device to load the data on ("cuda" or "cpu").
        group_name (str): test_MUA, train_MUA, or test_MUA_rep

    """

    subject_file_path = os.path.expanduser(f"~/Documents/BrainAlign_Data/THINGS_normMUA_{subject}.mat")

    # Load the data from the h5 file
    with h5py.File(subject_file_path, "r") as file:
        data_dict = {key: torch.tensor(np.array(file[key]), dtype=torch.float32, device=device) for key in file.keys()}
    
    # load data from dict
    object = data_dict[group_name]

    # remove noisy channels
    if snr_cutoff is not None:
        noise_mask = data_dict['SNR'].mean(axis=0) >= snr_cutoff
        if group_name == 'test_MUA_reps':
            object = object[:, :, noise_mask]
        else:
            object = object[:, noise_mask]
    
    roi_dict = {
        'V1': 1,
        'V4': 2,
        'IT': 3,
    }
    #     rois = {
    #     'monkeyF': {i: 'V1' for i in range(513)} | {i: 'V4' for i in range(833, 1024)} | {i: 'IT' for i in range(513, 833)},
    #     'monkeyN': {i: 'V1' for i in range(513)} | {i: 'V4' for i in range(513, 769)} | {i: 'IT' for i in range(769, 1024)},
    # }

    roi_mask = np.concatenate([[roi_dict['V1'] for i in range(513)], [roi_dict['V4'] for i in range(833, 1024)], [roi_dict['IT'] for i in range(513, 833)]])  if subject == 'F' else \
               np.concatenate([[roi_dict['V1'] for i in range(513)], [roi_dict['V4'] for i in range(513, 769)],  [roi_dict['IT'] for i in range(769, 1024)]]) if subject == 'N' else \
               None
    if snr_cutoff is not None:
        roi_mask = roi_mask[noise_mask]
    return object, roi_mask, roi_dict



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
    
    elif model_name == 'vit_b_16':

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)

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
    if dataset_name == 'ephys':
        
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


def extract_features_2(model, dataloader, device, return_nodes):

    # Apply return_nodes if specified
    if return_nodes:
        model = create_feature_extractor(model, return_nodes=return_nodes)

    model.eval()
    model.to(device)
    
    features = {layer: [] for layer in return_nodes.values()}
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting features"):
            inputs = batch[0].to(device)
            outputs = model(inputs)
            for layer, activation in outputs.items():
                # If activation is a tuple, take the first element
                if isinstance(activation, tuple):
                    activation = activation[0]
                # If activation is not a tensor, try to convert or skip
                if not isinstance(activation, torch.Tensor):
                    continue
                activation = torch.flatten(activation, start_dim=1)
                features[layer].append(activation.cpu())
    # Concatenate all batches
    for layer in features:
        if features[layer]:
            features[layer] = torch.cat(features[layer], dim=0)
        else:
            features[layer] = torch.empty(0)
    return features

############################ layer depth functions

def get_layer_depth(model_name, layer_name, normalize = False):
    
    layers = get_layer_order(model_name)
    layer_dict = dict(enumerate(layers))

    layer_depth = [key for key, value in layer_dict.items() if value == layer_name][0]
    
    if normalize:
        max = list(layer_dict.keys())[-1]
        layer_depth = layer_depth / max

    return layer_depth
    

def get_layer_order(model_name):

    if model_name == 'alexnet':
        layers= [
            'Conv1', 'Relu1', 'MaxPool1',
            'Conv2', 'Relu2', 'MaxPool2',
            'Conv3', 'Relu3',
            'Conv4', 'Relu4',
            'Conv5', 'Relu5', 'MaxPool5',
            'AvgPool', 'Dropout',
            'FullyConnected1', 'Relu6', 'Dropout2',
            'FullyConnected2', 'Relu7',
            'FullyConnected3',
        ]
        layer_dict = dict(enumerate(layers))
    elif model_name == 'resnet50':
        layers = [
            'conv1', 'bn1', 'relu', 'maxpool',
            'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.0.conv3', 'layer1.0.bn3', 'layer1.0.relu', 'layer1.0.downsample', 'layer1.0.downsample.0', 'layer1.0.downsample.1',
            'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.conv2', 'layer1.1.bn2', 'layer1.1.conv3', 'layer1.1.bn3', 'layer1.1.relu',
            'layer1.2', 'layer1.2.conv1', 'layer1.2.bn1', 'layer1.2.conv2', 'layer1.2.bn2', 'layer1.2.conv3', 'layer1.2.bn3', 'layer1.2.relu',
            'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.conv3', 'layer2.0.bn3', 'layer2.0.relu', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1',
            'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.conv2', 'layer2.1.bn2', 'layer2.1.conv3', 'layer2.1.bn3', 'layer2.1.relu',
            'layer2.2', 'layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.conv2', 'layer2.2.bn2', 'layer2.2.conv3', 'layer2.2.bn3', 'layer2.2.relu',
            'layer2.3', 'layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.conv2', 'layer2.3.bn2', 'layer2.3.conv3', 'layer2.3.bn3', 'layer2.3.relu',
            'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.conv3', 'layer3.0.bn3', 'layer3.0.relu', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1',
            'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.conv2', 'layer3.1.bn2', 'layer3.1.conv3', 'layer3.1.bn3', 'layer3.1.relu',
            'layer3.2', 'layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.conv2', 'layer3.2.bn2', 'layer3.2.conv3', 'layer3.2.bn3', 'layer3.2.relu',
            'layer3.3', 'layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.conv2', 'layer3.3.bn2', 'layer3.3.conv3', 'layer3.3.bn3', 'layer3.3.relu',
            'layer3.4', 'layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.conv2', 'layer3.4.bn2', 'layer3.4.conv3', 'layer3.4.bn3', 'layer3.4.relu',
            'layer3.5', 'layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.conv2', 'layer3.5.bn2', 'layer3.5.conv3', 'layer3.5.bn3', 'layer3.5.relu',
            'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.conv3', 'layer4.0.bn3', 'layer4.0.relu', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1',
            'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.conv2', 'layer4.1.bn2', 'layer4.1.conv3', 'layer4.1.bn3', 'layer4.1.relu',
            'layer4.2', 'layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.conv2', 'layer4.2.bn2', 'layer4.2.conv3', 'layer4.2.bn3', 'layer4.2.relu',
            'avgpool', 'fc'
        ]
        layer_dict = dict(enumerate(layers))
    elif model_name == 'vit_b_16':
        layers = [
            "conv_proj", "encoder", "encoder.dropout", "encoder.layers",

            "layer_0", 
            "layer_0.ln_1", "layer_0.self_attention", "layer_0.self_attention.out_proj", "layer_0.dropout", "layer_0.ln_2",
            "layer_0.mlp", "layer_0.mlp.0", "layer_0.mlp.1", "layer_0.mlp.2", "layer_0.mlp.3", "layer_0.mlp.4",
            "layer_1",
            "layer_1.ln_1", "layer_1.self_attention", "layer_1.self_attention.out_proj", "layer_1.dropout", "layer_1.ln_2",
            "layer_1.mlp", "layer_1.mlp.0", "layer_1.mlp.1", "layer_1.mlp.2", "layer_1.mlp.3", "layer_1.mlp.4",
            "layer_2",
            "layer_2.ln_1", "layer_2.self_attention", "layer_2.self_attention.out_proj", "layer_2.dropout", "layer_2.ln_2",
            "layer_2.mlp", "layer_2.mlp.0", "layer_2.mlp.1", "layer_2.mlp.2", "layer_2.mlp.3","layer_2.mlp.4",
            "layer_3",
            "layer_3.ln_1", "layer_3.self_attention", "layer_3.self_attention.out_proj", "layer_3.dropout", "layer_3.ln_2",
            "layer_3.mlp", "layer_3.mlp.0", "layer_3.mlp.1", "layer_3.mlp.2", "layer_3.mlp.3", "layer_3.mlp.4",
            "layer_4",
            "layer_4.ln_1", "layer_4.self_attention", "layer_4.self_attention.out_proj", "layer_4.dropout", "layer_4.ln_2",
            "layer_4.mlp", "layer_4.mlp.0", "layer_4.mlp.1", "layer_4.mlp.2", "layer_4.mlp.3", "layer_4.mlp.4", "layer_5",
            "layer_5",
            "layer_5.ln_1", "layer_5.self_attention", "layer_5.self_attention.out_proj", "layer_5.dropout", "layer_5.ln_2",
            "layer_5.mlp", "layer_5.mlp.0", "layer_5.mlp.1", "layer_5.mlp.2", "layer_5.mlp.3", "layer_5.mlp.4",
            "layer_6",
            "layer_6.ln_1", "layer_6.self_attention", "layer_6.self_attention.out_proj", "layer_6.dropout", "layer_6.ln_2",
            "layer_6.mlp", "layer_6.mlp.0", "layer_6.mlp.1", "layer_6.mlp.2", "layer_6.mlp.3", "layer_6.mlp.4",
            "layer_7",
            "layer_7.ln_1", "layer_7.self_attention", "layer_7.self_attention.out_proj", "layer_7.dropout", "layer_7.ln_2",
            "layer_7.mlp", "layer_7.mlp.0", "layer_7.mlp.1", "layer_7.mlp.2", "layer_7.mlp.3", "layer_7.mlp.4",
            "layer_8",
            "layer_8.ln_1", "layer_8.self_attention", "layer_8.self_attention.out_proj", "layer_8.dropout", "layer_8.ln_2",
            "layer_8.mlp", "layer_8.mlp.0", "layer_8.mlp.1", "layer_8.mlp.2", "layer_8.mlp.3","layer_8.mlp.4", 
            "layer_9",
            "layer_9.ln_1", "layer_9.self_attention", "layer_9.self_attention.out_proj", "layer_9.dropout", "layer_9.ln_2",
            "layer_9.mlp", "layer_9.mlp.0", "layer_9.mlp.1","layer_9.mlp.2", "layer_9.mlp.3", "layer_9.mlp.4",
            "layer_10",
            "layer_10.ln_1", "layer_10.self_attention", "layer_10.self_attention.out_proj", "layer_10.dropout", "layer_10.ln_2",
            "layer_10.mlp", "layer_10.mlp.0", "layer_10.mlp.1", "layer_10.mlp.2", "layer_10.mlp.3", "layer_10.mlp.4",
            "layer_11",
            "layer_11.ln_1", "layer_11.self_attention", "layer_11.self_attention.out_proj", "layer_11.dropout", "layer_11.ln_2",
            "layer_11.mlp", "layer_11.mlp.0", "layer_11.mlp.1", "layer_11.mlp.2", "layer_11.mlp.3", "layer_11.mlp.4",
            
            "layer_12.self_attention", # what is this?

            "encoder.ln", "heads", "heads.head"
        ]
    else:
        raise ValueError("Model not supported for layer depth calculation.")

    return layers
    
    
