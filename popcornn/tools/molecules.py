import numpy as np
import torch
import ase
from ase.io import read
from dataclasses import dataclass

from popcornn.tools.ase import pair_displacement


def process_images(raw_images, device):
    """
    Process the images.
    """
    if type(raw_images) == str:
        if raw_images.endswith('.npy'):
            raw_images = np.load(raw_images)
        elif raw_images.endswith('.pt'):
            raw_images = torch.load(raw_images)
        elif raw_images.endswith('.xyz'):
            raw_images = read(raw_images, index=':')
        else:
            raise ValueError(f"Cannot handle file type for {raw_images}.")
    
    assert len(raw_images) >= 2, "Must have at least two images."
    dtype = type(raw_images[0])
    if dtype is np.ndarray:
        raw_images = torch.tensor(raw_images, dtype=torch.float64)
        processed_images = Images(
            dtype=dtype,
            positions=raw_images,
            vec=raw_images[-1] - raw_images[0],
        )
    elif dtype is list:
        raw_images = torch.tensor(raw_images, dtype=torch.float64)
        processed_images = Images(
            dtype=dtype,
            positions=raw_images,
            vec=raw_images[-1] - raw_images[0],
        )
    elif dtype is torch.Tensor:
        processed_images = Images(
            dtype=dtype,
            positions=raw_images.float64(),
            vec=(raw_images[-1] - raw_images[0]).float64(),
        )
    elif issubclass(dtype, ase.Atoms):
        assert np.all(image.get_positions().shape == raw_images[0].get_positions().shape for image in raw_images), "All images must have the same shape."
        assert np.all(image.get_atomic_numbers() == raw_images[0].get_atomic_numbers() for image in raw_images), "All images must have the same atomic atomic_numbers."
        assert np.all(image.get_pbc() == raw_images[0].get_pbc() for image in raw_images), "All images must have the same pbc."
        assert np.all(image.get_cell() == raw_images[0].get_cell() for image in raw_images), "All images must have the same cell."
        assert np.all(image.get_tags() == raw_images[0].get_tags() for image in raw_images), "All images must have the same tags."
        processed_images = Images(
            dtype=dtype,
            positions=torch.tensor([image.get_positions().flatten() for image in raw_images], dtype=torch.float64),
            vec=torch.tensor(pair_displacement(raw_images[0], raw_images[-1]).flatten(), dtype=torch.float64),
            atomic_numbers=torch.tensor(raw_images[0].get_atomic_numbers(), dtype=torch.int64),
            pbc=torch.tensor(raw_images[0].get_pbc(), dtype=torch.bool),
            cell=torch.tensor(raw_images[0].get_cell(), dtype=torch.float64),
            tags=torch.tensor(raw_images[0].get_tags(), dtype=torch.int64),
        )
    else:
        raise ValueError(f"Cannot handle data type {dtype}.")
    
    processed_images = processed_images.to(device)
    return processed_images

        



