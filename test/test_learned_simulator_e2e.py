import os
import sys
import torch
import numpy as np
import tempfile

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gns.learned_simulator import LearnedSimulator


def make_simulator(architecture='sparse_egnn'):
    particle_dimensions = 2
    # encoder produces 10 velocity features + 4 boundary features = 14
    nnode_in = 14
    nedge_in = 3
    latent_dim = 64
    nmessage_passing_steps = 2
    nmlp_layers = 2
    mlp_hidden_dim = 64
    connectivity_radius = 1.0
    boundaries = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    normalization_stats = {
        "acceleration": {'mean': 0., 'std': 1.},
        "velocity": {'mean': 0., 'std': 1.}
    }
    nparticle_types = 1
    particle_type_embedding_size = 8
    boundary_clamp_limit = 1.0
    device = 'cpu'

    model = LearnedSimulator(
        particle_dimensions,
        nnode_in,
        nedge_in,
        latent_dim,
        nmessage_passing_steps,
        nmlp_layers,
        mlp_hidden_dim,
        connectivity_radius,
        boundaries,
        normalization_stats,
        nparticle_types,
        particle_type_embedding_size,
        boundary_clamp_limit,
        device,
        architecture=architecture,
    )
    return model


def test_learned_simulator_e2e_forward_backward():
    torch.manual_seed(0)
    model = make_simulator(architecture='sparse_egnn')
    model.train()

    nparticles = 10
    dim = 2
    # position_sequence shape (nparticles, 6, dim)
    position_sequence = torch.randn(nparticles, 6, dim)
    nparticles_per_example = torch.tensor([nparticles])
    particle_types = torch.zeros(nparticles, dtype=torch.long)
    position_sequence_noise = torch.zeros_like(position_sequence)

    # Run predict_accelerations (full pipeline) and compute loss
    predicted, target = model.predict_accelerations(
        next_positions=position_sequence[:, -1],
        position_sequence_noise=position_sequence_noise,
        position_sequence=position_sequence,
        nparticles_per_example=nparticles_per_example,
        particle_types=particle_types,
    )

    assert predicted.shape == (nparticles, dim)
    assert target.shape == (nparticles, dim)
    assert not torch.isnan(predicted).any()
    assert not torch.isnan(target).any()

    loss = (predicted - target).pow(2).mean()
    loss.backward()


def test_learned_simulator_save_load(tmp_path):
    torch.manual_seed(0)
    model = make_simulator(architecture='gns')
    model.eval()

    # small synthetic input
    nparticles = 6
    dim = 2
    position_sequence = torch.randn(nparticles, 6, dim)
    nparticles_per_example = torch.tensor([nparticles])
    particle_types = torch.zeros(nparticles, dtype=torch.long)

    out_before = model.predict_positions(position_sequence, nparticles_per_example, particle_types)

    # save to temporary file
    ckpt_file = tmp_path / "sim_checkpoint.pt"
    model.save(str(ckpt_file))

    # load payload and recreate model from saved config
    payload = torch.load(str(ckpt_file), map_location='cpu')
    assert 'state_dict' in payload
    assert 'config' in payload
    cfg = payload['config']

    # instantiate a fresh model from config and load weights
    new_model = LearnedSimulator(**cfg)
    new_model.load(str(ckpt_file))

    out_after = new_model.predict_positions(position_sequence, nparticles_per_example, particle_types)

    assert out_before.shape == out_after.shape
    assert not torch.isnan(out_after).any()
