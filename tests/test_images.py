from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read

from popcornn.tools import process_images
from tests import IMAGES_DIR


@pytest.mark.parametrize(
    "raw_images",
    [[[-0.558, 1.442], [-0.050, 0.467]], str(IMAGES_DIR / "muller_brown.json")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_list(raw_images, dtype, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(raw_images, device=device, dtype=dtype)
    assert images.image_type is list
    assert images.positions.shape == (2, 2)
    assert images.positions.device.type == device.type
    assert images.positions.dtype == dtype
    assert torch.allclose(
        images.positions,
        torch.tensor([[-0.558, 1.442], [-0.050, 0.467]], device=device, dtype=dtype),
    )
    assert images.fix_positions.shape == (2,)
    assert images.fix_positions.device.type == device.type
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(
        images.fix_positions == torch.zeros(2, device=device, dtype=torch.bool)
    )
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 2


@pytest.mark.parametrize(
    "raw_images",
    [
        np.array([[-0.558, 1.442], [-0.050, 0.467]]),
        str(IMAGES_DIR / "muller_brown.npy"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_numpy(raw_images, dtype, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(raw_images, device=device, dtype=dtype)
    assert images.image_type is np.ndarray
    assert images.positions.shape == (2, 2)
    assert images.positions.device.type == device.type
    assert images.positions.dtype == dtype
    assert torch.allclose(
        images.positions,
        torch.tensor([[-0.558, 1.442], [-0.050, 0.467]], device=device, dtype=dtype),
    )
    assert images.fix_positions.shape == (2,)
    assert images.fix_positions.device.type == device.type
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(
        images.fix_positions == torch.zeros(2, device=device, dtype=torch.bool)
    )
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 2


@pytest.mark.parametrize(
    "raw_images",
    [
        torch.tensor([[-0.558, 1.442], [-0.050, 0.467]]),
        str(IMAGES_DIR / "muller_brown.pt"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_torch(raw_images, dtype, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(raw_images, device=device, dtype=dtype)
    assert images.image_type is torch.Tensor
    assert images.positions.shape == (2, 2)
    assert images.positions.device.type == device.type
    assert images.positions.dtype == dtype
    assert torch.allclose(
        images.positions,
        torch.tensor([[-0.558, 1.442], [-0.050, 0.467]], device=device, dtype=dtype),
    )
    assert images.fix_positions.shape == (2,)
    assert images.fix_positions.device.type == device.type
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(
        images.fix_positions == torch.zeros(2, device=device, dtype=torch.bool)
    )
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 2


@pytest.mark.parametrize(
    "raw_images",
    [read(str(IMAGES_DIR / "OC20NEB.xyz"), index=":"), str(IMAGES_DIR / "OC20NEB.xyz")],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_xyz(raw_images, dtype, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(raw_images, device=device, dtype=dtype)
    assert images.image_type is Atoms
    assert images.positions.shape == (2, 42)
    assert images.positions.device.type == device.type
    assert images.positions.dtype == dtype
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    -0.0,
                    0.0,
                    12.65334702,
                    -0.0,
                    5.43758583,
                    15.82548046,
                    4.52291775,
                    2.71879292,
                    14.23941326,
                    4.52291775,
                    8.15637875,
                    12.0838213,
                    8.81918876,
                    0.22188029,
                    18.1014245,
                    8.4774802,
                    5.80893319,
                    21.04791724,
                    4.15724126,
                    2.98113633,
                    19.7234387,
                    4.52291775,
                    8.15637875,
                    17.41154671,
                    6.20783699,
                    3.01397571,
                    23.5278272,
                    7.13967253,
                    3.03131271,
                    24.12336599,
                    5.54306086,
                    3.82423157,
                    23.8682326,
                    5.70076617,
                    2.04184586,
                    23.67551037,
                    7.10627032,
                    2.5577278,
                    21.8160334,
                    6.48094153,
                    3.25790627,
                    22.12484476,
                ],
                [
                    -0.0,
                    0.0,
                    12.65334702,
                    -0.0,
                    5.43758583,
                    15.82548046,
                    4.52291775,
                    2.71879292,
                    14.23941326,
                    4.52291775,
                    8.15637875,
                    12.0838213,
                    8.80245269,
                    0.21060994,
                    18.10671856,
                    8.48235489,
                    5.79598315,
                    21.03052856,
                    4.11078265,
                    3.01817014,
                    19.72856215,
                    4.52291775,
                    8.15637875,
                    17.41154671,
                    6.21565284,
                    2.73169354,
                    26.50289708,
                    7.12057573,
                    2.75557713,
                    27.14212081,
                    5.55201927,
                    3.55544153,
                    26.81104362,
                    5.68376604,
                    1.77336548,
                    26.668565,
                    7.13484263,
                    2.24811624,
                    24.82548202,
                    6.5237351,
                    2.95101175,
                    25.11474215,
                ],
            ],
            device=device,
            dtype=dtype,
        ),
    )
    assert images.fix_positions.shape == (42,)
    assert images.fix_positions.device.type == device.type
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(
        images.fix_positions
        == torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            device=device,
            dtype=torch.bool,
        )
    )
    assert images.atomic_numbers is not None
    assert images.atomic_numbers.shape == (14,)
    assert images.atomic_numbers.device.type == device.type
    assert images.atomic_numbers.dtype == torch.long
    assert torch.all(
        images.atomic_numbers
        == torch.tensor(
            [55, 55, 55, 55, 55, 55, 55, 55, 6, 1, 1, 1, 1, 8],
            device=device,
            dtype=torch.long,
        )
    )
    assert images.pbc is not None
    assert images.pbc.shape == (3,)
    assert images.pbc.device.type == device.type
    assert images.pbc.dtype == torch.bool
    assert torch.all(images.pbc == torch.ones(3, device=device, dtype=torch.bool))
    assert images.cell is not None
    assert images.cell.shape == (3, 3)
    assert images.cell.device.type == device.type
    assert images.cell.dtype == dtype
    assert torch.allclose(
        images.cell,
        torch.tensor(
            [
                [9.04583549e00, 0.00000000e00, 5.53897714e-16],
                [-6.65912157e-16, 1.08751717e01, 1.01654017e00],
                [0.00000000e00, 0.00000000e00, 3.19663506e01],
            ],
            device=device,
            dtype=dtype,
        ),
    )
    assert images.tags is not None
    assert images.tags.shape == (14,)
    assert images.tags.device.type == device.type
    assert images.tags.dtype == torch.int
    assert torch.all(
        images.tags
        == torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2], device=device, dtype=torch.int
        )
    )
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge.device.type == device.type
    assert images.charge.dtype == torch.int
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin.device.type == device.type
    assert images.spin.dtype == torch.int
    assert images.spin == 0
    assert len(images) == 2


@pytest.mark.parametrize(
    "raw_images",
    [
        read(str(IMAGES_DIR / "OC20NEB.traj"), index=":"),
        str(IMAGES_DIR / "OC20NEB.traj"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_traj(raw_images, dtype, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(raw_images, device=device, dtype=dtype)
    assert images.image_type is Atoms
    assert images.positions.shape == (2, 42)
    assert images.positions.device.type == device.type
    assert images.positions.dtype == dtype
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    -0.0,
                    0.0,
                    12.65334702,
                    -0.0,
                    5.43758583,
                    15.82548046,
                    4.52291775,
                    2.71879292,
                    14.23941326,
                    4.52291775,
                    8.15637875,
                    12.0838213,
                    8.81918876,
                    0.22188029,
                    18.1014245,
                    8.4774802,
                    5.80893319,
                    21.04791724,
                    4.15724126,
                    2.98113633,
                    19.7234387,
                    4.52291775,
                    8.15637875,
                    17.41154671,
                    6.20783699,
                    3.01397571,
                    23.5278272,
                    7.13967253,
                    3.03131271,
                    24.12336599,
                    5.54306086,
                    3.82423157,
                    23.8682326,
                    5.70076617,
                    2.04184586,
                    23.67551037,
                    7.10627032,
                    2.5577278,
                    21.8160334,
                    6.48094153,
                    3.25790627,
                    22.12484476,
                ],
                [
                    -0.0,
                    0.0,
                    12.65334702,
                    -0.0,
                    5.43758583,
                    15.82548046,
                    4.52291775,
                    2.71879292,
                    14.23941326,
                    4.52291775,
                    8.15637875,
                    12.0838213,
                    8.80245269,
                    0.21060994,
                    18.10671856,
                    8.48235489,
                    5.79598315,
                    21.03052856,
                    4.11078265,
                    3.01817014,
                    19.72856215,
                    4.52291775,
                    8.15637875,
                    17.41154671,
                    6.21565284,
                    2.73169354,
                    26.50289708,
                    7.12057573,
                    2.75557713,
                    27.14212081,
                    5.55201927,
                    3.55544153,
                    26.81104362,
                    5.68376604,
                    1.77336548,
                    26.668565,
                    7.13484263,
                    2.24811624,
                    24.82548202,
                    6.5237351,
                    2.95101175,
                    25.11474215,
                ],
            ],
            device=device,
            dtype=dtype,
        ),
    )
    assert images.fix_positions.shape == (42,)
    assert images.fix_positions.device.type == device.type
    assert images.fix_positions.dtype == torch.bool
    assert torch.all(
        images.fix_positions
        == torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            device=device,
            dtype=torch.bool,
        )
    )
    assert images.atomic_numbers is not None
    assert images.atomic_numbers.shape == (14,)
    assert images.atomic_numbers.device.type == device.type
    assert images.atomic_numbers.dtype == torch.long
    assert torch.all(
        images.atomic_numbers
        == torch.tensor(
            [55, 55, 55, 55, 55, 55, 55, 55, 6, 1, 1, 1, 1, 8],
            device=device,
            dtype=torch.long,
        )
    )
    assert images.pbc is not None
    assert images.pbc.shape == (3,)
    assert images.pbc.device.type == device.type
    assert images.pbc.dtype == torch.bool
    assert torch.all(images.pbc == torch.ones(3, device=device, dtype=torch.bool))
    assert images.cell is not None
    assert images.cell.shape == (3, 3)
    assert images.cell.device.type == device.type
    assert images.cell.dtype == dtype
    assert torch.allclose(
        images.cell,
        torch.tensor(
            [
                [9.04583549e00, 0.00000000e00, 5.53897714e-16],
                [-6.65912157e-16, 1.08751717e01, 1.01654017e00],
                [0.00000000e00, 0.00000000e00, 3.19663506e01],
            ],
            device=device,
            dtype=dtype,
        ),
    )
    assert images.tags is not None
    assert images.tags.shape == (14,)
    assert images.tags.device.type == device.type
    assert images.tags.dtype == torch.int
    assert torch.all(
        images.tags
        == torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2], device=device, dtype=torch.int
        )
    )
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge.device.type == device.type
    assert images.charge.dtype == torch.int
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin.device.type == device.type
    assert images.spin.dtype == torch.int
    assert images.spin == 0
    assert len(images) == 2


@pytest.mark.parametrize(
    "raw_images",
    [
        [[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]],
        str(IMAGES_DIR / "muller_brown_all.json"),
    ],
)
def test_list_with_intermediates(raw_images):
    images = process_images(raw_images, device=torch.device("cpu"), dtype=torch.float32)
    assert images.image_type is list
    assert images.positions.shape == (3, 2)
    assert torch.allclose(
        images.positions,
        torch.tensor([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]]),
    )
    assert images.fix_positions.shape == (2,)
    assert torch.all(images.fix_positions == torch.zeros(2, dtype=torch.bool))
    assert images.atomic_numbers is None
    assert images.pbc is None
    assert images.cell is None
    assert images.tags is None
    assert images.charge is None
    assert images.spin is None
    assert len(images) == 3


@pytest.mark.parametrize(
    "raw_images",
    [
        read(str(IMAGES_DIR / "OC20NEB_all.xyz"), index=":"),
        str(IMAGES_DIR / "OC20NEB_all.xyz"),
    ],
)
def test_xyz_with_intermediates(raw_images):
    images = process_images(raw_images, device=torch.device("cpu"), dtype=torch.float32)
    assert images.image_type is Atoms
    assert images.positions.shape == (10, 42)
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.81918876e00,
                    2.21880285e-01,
                    1.81014245e01,
                    8.47748020e00,
                    5.80893319e00,
                    2.10479172e01,
                    4.15724126e00,
                    2.98113633e00,
                    1.97234387e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.20783699e00,
                    3.01397571e00,
                    2.35278272e01,
                    7.13967253e00,
                    3.03131271e00,
                    2.41233660e01,
                    5.54306086e00,
                    3.82423157e00,
                    2.38682326e01,
                    5.70076617e00,
                    2.04184586e00,
                    2.36755104e01,
                    7.10627032e00,
                    2.55772780e00,
                    2.18160334e01,
                    6.48094153e00,
                    3.25790627e00,
                    2.21248448e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.86491061e00,
                    2.84494583e-01,
                    1.81445380e01,
                    8.46617697e00,
                    5.75445760e00,
                    2.11690355e01,
                    4.20406762e00,
                    2.96850271e00,
                    1.98827589e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.21925077e00,
                    2.98210172e00,
                    2.38625663e01,
                    7.16356265e00,
                    3.00976260e00,
                    2.44375078e01,
                    5.55525247e00,
                    3.77731140e00,
                    2.42357552e01,
                    5.72745355e00,
                    2.00193796e00,
                    2.40146754e01,
                    7.08780168e00,
                    2.57865129e00,
                    2.21199307e01,
                    6.45244096e00,
                    3.25444780e00,
                    2.24574392e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.86153388e00,
                    2.90387709e-01,
                    1.81424779e01,
                    8.43697729e00,
                    5.69237218e00,
                    2.12535921e01,
                    4.23259150e00,
                    2.91552698e00,
                    1.99615919e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.21974607e00,
                    2.96717079e00,
                    2.41993237e01,
                    7.16072527e00,
                    2.97888007e00,
                    2.47806605e01,
                    5.56010826e00,
                    3.76392243e00,
                    2.45780156e01,
                    5.71988902e00,
                    1.98956967e00,
                    2.43410772e01,
                    7.08357050e00,
                    2.56645238e00,
                    2.24554430e01,
                    6.46079946e00,
                    3.24939027e00,
                    2.27978230e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.85086634e00,
                    2.93998580e-01,
                    1.81631468e01,
                    8.43360372e00,
                    5.66207995e00,
                    2.13220709e01,
                    4.23528017e00,
                    2.87022147e00,
                    1.99832971e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.23258466e00,
                    2.94056520e00,
                    2.45334561e01,
                    7.16576711e00,
                    2.95758583e00,
                    2.51285637e01,
                    5.57067210e00,
                    3.74052122e00,
                    2.49005699e01,
                    5.72891552e00,
                    1.96471071e00,
                    2.46761889e01,
                    7.06300793e00,
                    2.49879235e00,
                    2.27875590e01,
                    6.48599054e00,
                    3.21578787e00,
                    2.31339583e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.83955477e00,
                    3.06764616e-01,
                    1.81331944e01,
                    8.44701300e00,
                    5.66826866e00,
                    2.13156700e01,
                    4.22464497e00,
                    2.87259865e00,
                    1.99253776e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.23702732e00,
                    2.89956241e00,
                    2.48607867e01,
                    7.16164980e00,
                    2.91703441e00,
                    2.54693665e01,
                    5.57465199e00,
                    3.70695650e00,
                    2.52125765e01,
                    5.72749812e00,
                    1.92820446e00,
                    2.50131012e01,
                    7.06877282e00,
                    2.42614850e00,
                    2.31248277e01,
                    6.50638338e00,
                    3.15533186e00,
                    2.34636008e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.83655560e00,
                    3.07389807e-01,
                    1.81368239e01,
                    8.45576444e00,
                    5.68006453e00,
                    2.13194581e01,
                    4.20670918e00,
                    2.85452913e00,
                    1.98812463e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.24120963e00,
                    2.87207926e00,
                    2.52014449e01,
                    7.17031630e00,
                    2.89483677e00,
                    2.58038056e01,
                    5.58132576e00,
                    3.68112164e00,
                    2.55535292e01,
                    5.73125644e00,
                    1.90175181e00,
                    2.53623762e01,
                    7.06440106e00,
                    2.40165189e00,
                    2.34631444e01,
                    6.49792042e00,
                    3.12446294e00,
                    2.38029024e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.82799649e00,
                    2.98032915e-01,
                    1.81146671e01,
                    8.47960631e00,
                    5.70336701e00,
                    2.12855725e01,
                    4.19441169e00,
                    2.85261782e00,
                    1.98360309e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.24177176e00,
                    2.84514910e00,
                    2.55559467e01,
                    7.16726888e00,
                    2.85102273e00,
                    2.61648215e01,
                    5.58112319e00,
                    3.64953860e00,
                    2.59178239e01,
                    5.72799994e00,
                    1.87485273e00,
                    2.57031954e01,
                    7.06390461e00,
                    2.39709323e00,
                    2.38125245e01,
                    6.50461418e00,
                    3.11774430e00,
                    2.41639952e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.82115436e00,
                    2.89475600e-01,
                    1.80962124e01,
                    8.52111067e00,
                    5.74246566e00,
                    2.12400228e01,
                    4.18910331e00,
                    2.85030466e00,
                    1.97836435e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.23622098e00,
                    2.80914793e00,
                    2.59233610e01,
                    7.16035144e00,
                    2.80895959e00,
                    2.65342123e01,
                    5.57317626e00,
                    3.60834340e00,
                    2.62929044e01,
                    5.72442831e00,
                    1.83552366e00,
                    2.60565735e01,
                    7.05571815e00,
                    2.38485823e00,
                    2.41742361e01,
                    6.50061551e00,
                    3.10201935e00,
                    2.45380237e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.81917382e00,
                    2.80714940e-01,
                    1.80992311e01,
                    8.54527344e00,
                    5.76748674e00,
                    2.12032472e01,
                    4.17566636e00,
                    2.85850270e00,
                    1.97507938e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.23309175e00,
                    2.77641132e00,
                    2.62765055e01,
                    7.15905818e00,
                    2.77500157e00,
                    2.68849723e01,
                    5.56930396e00,
                    3.57379200e00,
                    2.66475955e01,
                    5.72132246e00,
                    1.80251514e00,
                    2.64075818e01,
                    7.06157166e00,
                    2.36553046e00,
                    2.45311719e01,
                    6.49470487e00,
                    3.07287893e00,
                    2.48921559e01,
                ],
                [
                    -1.63483589e-33,
                    2.66988983e-17,
                    1.26533470e01,
                    -3.32956078e-16,
                    5.43758583e00,
                    1.58254805e01,
                    4.52291775e00,
                    2.71879292e00,
                    1.42394133e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.20838213e01,
                    8.80245269e00,
                    2.10609936e-01,
                    1.81067186e01,
                    8.48235489e00,
                    5.79598315e00,
                    2.10305286e01,
                    4.11078265e00,
                    3.01817014e00,
                    1.97285622e01,
                    4.52291775e00,
                    8.15637875e00,
                    1.74115467e01,
                    6.21565284e00,
                    2.73169354e00,
                    2.65028971e01,
                    7.12057573e00,
                    2.75557713e00,
                    2.71421208e01,
                    5.55201927e00,
                    3.55544153e00,
                    2.68110436e01,
                    5.68376604e00,
                    1.77336548e00,
                    2.66685650e01,
                    7.13484263e00,
                    2.24811624e00,
                    2.48254820e01,
                    6.52373510e00,
                    2.95101175e00,
                    2.51147421e01,
                ],
            ]
        ),
    )
    assert images.fix_positions.shape == (42,)
    assert torch.all(
        images.fix_positions
        == torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
        )
    )
    assert images.fix_positions.shape == (42,)
    assert torch.all(
        images.fix_positions
        == torch.tensor(
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
        )
    )
    assert images.atomic_numbers is not None
    assert images.atomic_numbers.shape == (14,)
    assert torch.all(
        images.atomic_numbers
        == torch.tensor(
            [55, 55, 55, 55, 55, 55, 55, 55, 6, 1, 1, 1, 1, 8], dtype=torch.long
        )
    )
    assert images.pbc is not None
    assert images.pbc.shape == (3,)
    assert torch.all(images.pbc == torch.ones(3, dtype=torch.bool))
    assert images.cell is not None
    assert images.cell.shape == (3, 3)
    assert torch.allclose(
        images.cell,
        torch.tensor(
            [
                [9.04583549e00, 0.00000000e00, 5.53897714e-16],
                [-6.65912157e-16, 1.08751717e01, 1.01654017e00],
                [0.00000000e00, 0.00000000e00, 3.19663506e01],
            ]
        ),
    )
    assert images.tags is not None
    assert images.tags.shape == (14,)
    assert torch.all(
        images.tags
        == torch.tensor([0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2], dtype=torch.int)
    )
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin == 0
    assert len(images) == 10


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda")])
def test_charge_spin(device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="CUDA is not available, skipping test.")

    images = process_images(
        str(IMAGES_DIR / "T1x.xyz"), device=device, dtype=torch.float32
    )
    assert images.charge is not None
    assert images.charge.shape == ()
    assert images.charge.device.type == device.type
    assert images.charge.dtype == torch.int
    assert images.charge == 0
    assert images.spin is not None
    assert images.spin.shape == ()
    assert images.spin.device.type == device.type
    assert images.spin.dtype == torch.int
    assert images.spin == 1


def test_empty():
    with pytest.raises(AssertionError, match=r"Must have at least two images\."):
        process_images([], device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"Cannot handle file type for invalid\.txt\."):
        process_images("invalid.txt", device=torch.device("cpu"), dtype=torch.float32)


def test_unwrap(caplog):
    UNWRAP_WARNING = "Unwrapping atom positions. Assuming no atoms move more than half the box length."
    NO_UNWRAP_WARNING = "Not unwrapping atom positions. Assuming atoms are already unwrapped or do not travel across period boundaries."

    caplog.clear()
    images = process_images(
        str(IMAGES_DIR / "LJ35.xyz"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unwrap_positions=True,
    )
    assert UNWRAP_WARNING in caplog.text
    assert NO_UNWRAP_WARNING not in caplog.text
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    0.56123102,
                    0.32402688,
                    0.45824321,
                    0.0,
                    0.64805377,
                    3.20770249,
                    0.56123102,
                    0.32402688,
                    2.29121606,
                    -0.56123102,
                    1.62013441,
                    1.37472964,
                    0.0,
                    1.29610753,
                    0.45824321,
                    -0.56123102,
                    1.62013441,
                    3.20770249,
                    0.0,
                    1.29610753,
                    2.29121606,
                    -1.12246205,
                    2.59221506,
                    1.37472964,
                    -0.56123102,
                    2.26818818,
                    0.45824321,
                    -1.12246205,
                    2.59221506,
                    3.20770249,
                    -0.56123102,
                    2.26818818,
                    2.29121606,
                    1.12246205,
                    0.64805377,
                    1.37472964,
                    1.68369307,
                    0.32402688,
                    0.45824321,
                    1.12246205,
                    0.64805377,
                    3.20770249,
                    1.68369307,
                    0.32402688,
                    2.29121606,
                    0.56123102,
                    1.62013441,
                    1.37472964,
                    1.12246205,
                    1.29610753,
                    0.45824321,
                    0.56123102,
                    1.62013441,
                    3.20770249,
                    1.12246205,
                    1.29610753,
                    2.29121606,
                    0.0,
                    2.59221506,
                    1.37472964,
                    0.56123102,
                    2.26818818,
                    0.45824321,
                    0.0,
                    2.59221506,
                    3.20770249,
                    0.56123102,
                    2.26818818,
                    2.29121606,
                    2.2449241,
                    0.64805377,
                    1.37472964,
                    2.80615512,
                    0.32402688,
                    0.45824321,
                    2.2449241,
                    0.64805377,
                    3.20770249,
                    2.80615512,
                    0.32402688,
                    2.29121606,
                    1.68369307,
                    1.62013441,
                    1.37472964,
                    2.2449241,
                    1.29610753,
                    0.45824321,
                    1.68369307,
                    1.62013441,
                    3.20770249,
                    2.2449241,
                    1.29610753,
                    2.29121606,
                    1.12246205,
                    2.59221506,
                    1.37472964,
                    1.68369307,
                    2.26818818,
                    0.45824321,
                    1.12246205,
                    2.59221506,
                    3.20770249,
                    1.68369307,
                    2.26818818,
                    2.29121606,
                ],
                [
                    0.56123102,
                    0.32402688,
                    0.45824321,
                    0.0,
                    0.64805377,
                    3.20770249,
                    0.56123102,
                    0.32402688,
                    2.29121606,
                    -0.56123102,
                    1.62013441,
                    1.37472964,
                    0.0,
                    1.29610753,
                    0.45824321,
                    -0.56123102,
                    1.62013441,
                    3.20770249,
                    0.0,
                    1.29610753,
                    2.29121606,
                    -1.68369307,
                    3.56429572,
                    1.37472964,
                    -0.56123102,
                    2.26818818,
                    0.45824321,
                    -1.12246205,
                    2.59221506,
                    3.20770249,
                    -0.56123102,
                    2.26818818,
                    2.29121606,
                    1.12246205,
                    0.64805377,
                    1.37472964,
                    1.68369307,
                    0.32402688,
                    0.45824321,
                    1.12246205,
                    0.64805377,
                    3.20770249,
                    1.68369307,
                    0.32402688,
                    2.29121606,
                    0.56123102,
                    1.62013441,
                    1.37472964,
                    1.12246205,
                    1.29610753,
                    0.45824321,
                    0.56123102,
                    1.62013441,
                    3.20770249,
                    1.12246205,
                    1.29610753,
                    2.29121606,
                    0.0,
                    2.59221506,
                    1.37472964,
                    0.56123102,
                    2.26818818,
                    0.45824321,
                    0.0,
                    2.59221506,
                    3.20770249,
                    0.56123102,
                    2.26818818,
                    2.29121606,
                    2.2449241,
                    0.64805377,
                    1.37472964,
                    2.80615512,
                    0.32402688,
                    0.45824321,
                    2.2449241,
                    0.64805377,
                    3.20770249,
                    2.80615512,
                    0.32402688,
                    2.29121606,
                    1.68369307,
                    1.62013441,
                    1.37472964,
                    2.2449241,
                    1.29610753,
                    0.45824321,
                    1.68369307,
                    1.62013441,
                    3.20770249,
                    2.2449241,
                    1.29610753,
                    2.29121606,
                    1.12246205,
                    2.59221506,
                    1.37472964,
                    1.68369307,
                    2.26818818,
                    0.45824321,
                    1.12246205,
                    2.59221506,
                    3.20770249,
                    1.68369307,
                    2.26818818,
                    2.29121606,
                ],
            ],
            dtype=torch.float32,
        ),
    )

    caplog.clear()
    images = process_images(
        str(IMAGES_DIR / "LJ35.xyz"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unwrap_positions=False,
    )
    assert UNWRAP_WARNING not in caplog.text
    assert NO_UNWRAP_WARNING in caplog.text
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    0.56123102,
                    0.32402688,
                    0.45824321,
                    0.0,
                    0.64805377,
                    3.20770249,
                    0.56123102,
                    0.32402688,
                    2.29121606,
                    -0.56123102,
                    1.62013441,
                    1.37472964,
                    0.0,
                    1.29610753,
                    0.45824321,
                    -0.56123102,
                    1.62013441,
                    3.20770249,
                    0.0,
                    1.29610753,
                    2.29121606,
                    -1.12246205,
                    2.59221506,
                    1.37472964,
                    -0.56123102,
                    2.26818818,
                    0.45824321,
                    -1.12246205,
                    2.59221506,
                    3.20770249,
                    -0.56123102,
                    2.26818818,
                    2.29121606,
                    1.12246205,
                    0.64805377,
                    1.37472964,
                    1.68369307,
                    0.32402688,
                    0.45824321,
                    1.12246205,
                    0.64805377,
                    3.20770249,
                    1.68369307,
                    0.32402688,
                    2.29121606,
                    0.56123102,
                    1.62013441,
                    1.37472964,
                    1.12246205,
                    1.29610753,
                    0.45824321,
                    0.56123102,
                    1.62013441,
                    3.20770249,
                    1.12246205,
                    1.29610753,
                    2.29121606,
                    0.0,
                    2.59221506,
                    1.37472964,
                    0.56123102,
                    2.26818818,
                    0.45824321,
                    0.0,
                    2.59221506,
                    3.20770249,
                    0.56123102,
                    2.26818818,
                    2.29121606,
                    2.2449241,
                    0.64805377,
                    1.37472964,
                    2.80615512,
                    0.32402688,
                    0.45824321,
                    2.2449241,
                    0.64805377,
                    3.20770249,
                    2.80615512,
                    0.32402688,
                    2.29121606,
                    1.68369307,
                    1.62013441,
                    1.37472964,
                    2.2449241,
                    1.29610753,
                    0.45824321,
                    1.68369307,
                    1.62013441,
                    3.20770249,
                    2.2449241,
                    1.29610753,
                    2.29121606,
                    1.12246205,
                    2.59221506,
                    1.37472964,
                    1.68369307,
                    2.26818818,
                    0.45824321,
                    1.12246205,
                    2.59221506,
                    3.20770249,
                    1.68369307,
                    2.26818818,
                    2.29121606,
                ],
                [
                    0.56123102,
                    0.32402688,
                    0.45824321,
                    0.0,
                    0.64805377,
                    3.20770249,
                    0.56123102,
                    0.32402688,
                    2.29121606,
                    -0.56123102,
                    1.62013441,
                    1.37472964,
                    0.0,
                    1.29610753,
                    0.45824321,
                    -0.56123102,
                    1.62013441,
                    3.20770249,
                    0.0,
                    1.29610753,
                    2.29121606,
                    0.0,
                    0.64805377,
                    1.37472964,
                    -0.56123102,
                    2.26818818,
                    0.45824321,
                    -1.12246205,
                    2.59221506,
                    3.20770249,
                    -0.56123102,
                    2.26818818,
                    2.29121606,
                    1.12246205,
                    0.64805377,
                    1.37472964,
                    1.68369307,
                    0.32402688,
                    0.45824321,
                    1.12246205,
                    0.64805377,
                    3.20770249,
                    1.68369307,
                    0.32402688,
                    2.29121606,
                    0.56123102,
                    1.62013441,
                    1.37472964,
                    1.12246205,
                    1.29610753,
                    0.45824321,
                    0.56123102,
                    1.62013441,
                    3.20770249,
                    1.12246205,
                    1.29610753,
                    2.29121606,
                    0.0,
                    2.59221506,
                    1.37472964,
                    0.56123102,
                    2.26818818,
                    0.45824321,
                    0.0,
                    2.59221506,
                    3.20770249,
                    0.56123102,
                    2.26818818,
                    2.29121606,
                    2.2449241,
                    0.64805377,
                    1.37472964,
                    2.80615512,
                    0.32402688,
                    0.45824321,
                    2.2449241,
                    0.64805377,
                    3.20770249,
                    2.80615512,
                    0.32402688,
                    2.29121606,
                    1.68369307,
                    1.62013441,
                    1.37472964,
                    2.2449241,
                    1.29610753,
                    0.45824321,
                    1.68369307,
                    1.62013441,
                    3.20770249,
                    2.2449241,
                    1.29610753,
                    2.29121606,
                    1.12246205,
                    2.59221506,
                    1.37472964,
                    1.68369307,
                    2.26818818,
                    0.45824321,
                    1.12246205,
                    2.59221506,
                    3.20770249,
                    1.68369307,
                    2.26818818,
                    2.29121606,
                ],
            ],
            dtype=torch.float32,
        ),
    )

    caplog.clear()
    images = process_images(
        str(IMAGES_DIR / "LJ13.xyz"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unwrap_positions=True,
    )
    assert UNWRAP_WARNING not in caplog.text
    assert NO_UNWRAP_WARNING not in caplog.text
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    0.34918998,
                    -0.8434986,
                    0.58047443,
                    -0.63106228,
                    -0.87870054,
                    0.00445351,
                    0.3579967,
                    -0.85558812,
                    -0.55693933,
                    -0.53220336,
                    -0.20878949,
                    0.91844467,
                    0.51795379,
                    0.22835074,
                    0.92192944,
                    -0.3579967,
                    0.85558812,
                    0.55693933,
                    -0.51795379,
                    -0.22835074,
                    -0.92192944,
                    1.06812768,
                    -0.17139279,
                    0.01009198,
                    -1.06812768,
                    0.17139279,
                    -0.01009198,
                    0.53220336,
                    0.20878949,
                    -0.91844467,
                    -0.34918998,
                    0.8434986,
                    -0.58047443,
                    0.63106228,
                    0.87870054,
                    -0.00445351,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.22835074,
                    -0.51795379,
                    0.92192944,
                    -0.17139279,
                    -1.06812768,
                    0.01009198,
                    0.20878949,
                    -0.53220336,
                    -0.91844467,
                    -0.8434986,
                    -0.34918998,
                    0.58047443,
                    0.85558812,
                    0.3579967,
                    0.55693933,
                    -0.20878949,
                    0.53220336,
                    0.91844467,
                    -0.85558812,
                    -0.3579967,
                    -0.55693933,
                    0.87870054,
                    -0.63106228,
                    -0.00445351,
                    -0.87870054,
                    0.63106228,
                    0.00445351,
                    0.8434986,
                    0.34918998,
                    -0.58047443,
                    -0.22835074,
                    0.51795379,
                    -0.92192944,
                    0.17139279,
                    1.06812768,
                    -0.01009198,
                    0.0,
                    -0.0,
                    0.0,
                ],
            ],
            dtype=torch.float32,
        ),
    )

    caplog.clear()
    images = process_images(
        str(IMAGES_DIR / "LJ13.xyz"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        unwrap_positions=False,
    )
    assert UNWRAP_WARNING not in caplog.text
    assert NO_UNWRAP_WARNING not in caplog.text
    assert torch.allclose(
        images.positions,
        torch.tensor(
            [
                [
                    0.34918998,
                    -0.8434986,
                    0.58047443,
                    -0.63106228,
                    -0.87870054,
                    0.00445351,
                    0.3579967,
                    -0.85558812,
                    -0.55693933,
                    -0.53220336,
                    -0.20878949,
                    0.91844467,
                    0.51795379,
                    0.22835074,
                    0.92192944,
                    -0.3579967,
                    0.85558812,
                    0.55693933,
                    -0.51795379,
                    -0.22835074,
                    -0.92192944,
                    1.06812768,
                    -0.17139279,
                    0.01009198,
                    -1.06812768,
                    0.17139279,
                    -0.01009198,
                    0.53220336,
                    0.20878949,
                    -0.91844467,
                    -0.34918998,
                    0.8434986,
                    -0.58047443,
                    0.63106228,
                    0.87870054,
                    -0.00445351,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.22835074,
                    -0.51795379,
                    0.92192944,
                    -0.17139279,
                    -1.06812768,
                    0.01009198,
                    0.20878949,
                    -0.53220336,
                    -0.91844467,
                    -0.8434986,
                    -0.34918998,
                    0.58047443,
                    0.85558812,
                    0.3579967,
                    0.55693933,
                    -0.20878949,
                    0.53220336,
                    0.91844467,
                    -0.85558812,
                    -0.3579967,
                    -0.55693933,
                    0.87870054,
                    -0.63106228,
                    -0.00445351,
                    -0.87870054,
                    0.63106228,
                    0.00445351,
                    0.8434986,
                    0.34918998,
                    -0.58047443,
                    -0.22835074,
                    0.51795379,
                    -0.92192944,
                    0.17139279,
                    1.06812768,
                    -0.01009198,
                    0.0,
                    -0.0,
                    0.0,
                ],
            ],
            dtype=torch.float32,
        ),
    )
