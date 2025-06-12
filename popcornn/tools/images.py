import numpy as np
import torch
import ase
from ase.io import read
from ase.constraints import FixAtoms
from dataclasses import dataclass


@dataclass
class Images():
    """
    Data class representing the images.

    Attributes:
    -----------
    image_type: type
        The data type of the images.
    positions: torch.Tensor
        The positions of the images.
    atomic_numbers: torch.Tensor, optional
        The atomic atomic_numbers of the images.
    pbc: torch.Tensor, optional
        The periodic boundary conditions of the images.
    cell: torch.Tensor, optional
        The cell dimensions of the images.
    tags: torch.Tensor, optional
        The tags of the atoms in the images.
    """
    image_type: type
    positions: torch.Tensor
    fix_positions: torch.Tensor
    atomic_numbers: torch.Tensor = None
    pbc: torch.Tensor = None
    cell: torch.Tensor = None
    tags: torch.Tensor = None
    charge: torch.Tensor = None
    spin: torch.Tensor = None

    def __len__(self):
        """
        Return the number of images.
        """
        return len(self.positions)

    def to(self, device):
        """
        Move the images to the specified device.
        """
        self.positions = self.positions.to(device)
        if self.atomic_numbers is not None:
            self.atomic_numbers = self.atomic_numbers.to(device)
        if self.pbc is not None:
            self.pbc = self.pbc.to(device)
        if self.cell is not None:
            self.cell = self.cell.to(device)
        if self.fix_positions is not None:
            self.fix_positions = self.fix_positions.to(device)
        if self.tags is not None:
            self.tags = self.tags.to(device)
        if self.charge is not None:
            self.charge = self.charge.to(device)
        if self.spin is not None:
            self.spin = self.spin.to(device)
        return self


def process_images(raw_images, device, dtype):
    """
    Process the images.
    """
    if type(raw_images) == str:
        if raw_images.endswith('.npy'):
            raw_images = np.load(raw_images)
        elif raw_images.endswith('.pt'):
            raw_images = torch.load(raw_images)
        elif raw_images.endswith('.xyz') or raw_images.endswith('.traj'):
            raw_images = read(raw_images, index=':')
        else:
            raise ValueError(f"Cannot handle file type for {raw_images}.")
    
    assert len(raw_images) >= 2, "Must have at least two images."
    image_type = type(raw_images[0])
    if image_type in [np.ndarray, list]:
        raw_images = torch.tensor(raw_images, device=device, dtype=dtype)
        fix_positions = torch.zeros_like(raw_images, dtype=torch.bool)
        processed_images = Images(
            image_type=image_type,
            positions=raw_images,
            fix_positions=fix_positions,
        )
    elif image_type is torch.Tensor:
        raw_images = raw_images.to(device=device, dtype=dtype)
        fix_positions = torch.zeros_like(raw_images, dtype=torch.bool)
        processed_images = Images(
            image_type=image_type,
            positions=raw_images,
            fix_positions=fix_positions,
        )
    elif issubclass(image_type, ase.Atoms):
        assert np.all(image.get_positions().shape == raw_images[0].get_positions().shape for image in raw_images), "All images must have the same shape."
        positions = torch.tensor([image.get_positions().flatten() for image in raw_images], device=device, dtype=dtype)
        assert np.all(image.get_atomic_numbers() == raw_images[0].get_atomic_numbers() for image in raw_images), "All images must have the same atomic atomic_numbers."
        atomic_numbers = torch.tensor(raw_images[0].get_atomic_numbers(), device=device, dtype=torch.int)
        assert np.all(image.get_pbc() == raw_images[0].get_pbc() for image in raw_images), "All images must have the same pbc."
        pbc = torch.tensor(raw_images[0].get_pbc(), device=device, dtype=torch.bool)
        assert np.all(image.get_cell() == raw_images[0].get_cell() for image in raw_images), "All images must have the same cell."
        cell = torch.tensor(raw_images[0].get_cell().array, device=device, dtype=torch.float)
        assert np.all(image.constraints.__repr__() == raw_images[0].constraints.__repr__() for image in raw_images), "All images must have the same constraints."
        fix_positions = torch.zeros_like(positions[0], dtype=torch.bool)
        fix_positions = fix_positions.view(-1, 3)
        for constraint in raw_images[0].constraints:
            if isinstance(constraint, FixAtoms):
                fix_positions[constraint.index] = True
            else:
                raise ValueError(f"Cannot handle constraint type {type(constraint)}.")
        fix_positions = fix_positions.flatten()
        assert np.all(image.get_tags() == raw_images[0].get_tags() for image in raw_images), "All images must have the same tags."
        tags = torch.tensor(raw_images[0].get_tags(), device=device, dtype=torch.int)
        assert np.all(image.info.get('charge', 0) == raw_images[0].info.get('charge', 0) for image in raw_images), "All images must have the same charge."
        charge = torch.tensor(raw_images[0].info.get('charge', 0), device=device, dtype=torch.int)
        assert np.all(image.info.get('spin', 0) == raw_images[0].info.get('spin', 0) for image in raw_images), "All images must have the same spin."
        spin = torch.tensor(raw_images[0].info.get('spin', 0), device=device, dtype=torch.int)
        processed_images = Images(
            image_type=image_type,
            positions=positions,
            fix_positions=fix_positions,
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            cell=cell,
            tags=tags,
            charge=charge,
            spin=spin,
        )
    else:
        raise ValueError(f"Cannot handle data type {dtype}.")
    
    processed_images = processed_images.to(device)
    return processed_images

        



