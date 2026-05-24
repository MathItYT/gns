import os
import sys

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from gns import graph_network


def main():
  torch.manual_seed(0)

  n_particles = 8
  dim = 2
  nnode_in_features = 30
  nedge_in_features = 3
  latent_dim = 128
  n_edges = 20

  edge_index = torch.randint(0, n_particles, (2, n_edges), dtype=torch.long)
  node_features = torch.randn(n_particles, nnode_in_features)
  coors = torch.randn(n_particles, dim)
  edge_features = torch.randn(n_edges, nedge_in_features)

  model = graph_network.EncodeProcessDecodeSparseEGNN(
      nnode_in_features=nnode_in_features,
      nnode_out_features=dim,
      nedge_in_features=nedge_in_features,
      latent_dim=latent_dim,
      nmessage_passing_steps=2,
      nmlp_layers=2,
      mlp_hidden_dim=latent_dim,
      connectivity_radius=1.0,
  )

  out = model(node_features, coors, edge_index, edge_features)
  assert out.shape == (n_particles, dim), (
      f"Expected output shape {(n_particles, dim)}, got {tuple(out.shape)}")
  assert not torch.isnan(out).any(), "SparseEGNN output contains NaNs"

  loss = out.pow(2).mean()
  loss.backward()

  print("SparseEGNN forward/backward test passed.")


if __name__ == "__main__":
  main()
