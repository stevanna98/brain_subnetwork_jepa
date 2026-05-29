"""Tests for dataset output shapes and graph construction."""

import pytest
import torch
from torch_geometric.data import Data

from brain_jepa.data.atlas import load_atlas
from brain_jepa.data.connectivity import build_graph, pearson_correlation
from brain_jepa.data.dataset import SyntheticBrainDataset

ATLAS_CSV = "data/atlas/glasser379_to_rsn12.csv"
N = 379
F = 32


@pytest.fixture(scope="module")
def atlas():
    return load_atlas(ATLAS_CSV)


@pytest.fixture(scope="module")
def dataset(atlas):
    return SyntheticBrainDataset(atlas=atlas, num_subjects=4, feature_dim=F, seed=1)


def test_dataset_length(dataset):
    assert len(dataset) == 4


def test_item_type(dataset):
    item = dataset[0]
    assert isinstance(item, Data)


def test_node_features_shape(dataset):
    item = dataset[0]
    assert item.x.shape == (N, F)


def test_rsn_ids_shape(dataset):
    item = dataset[0]
    assert item.rsn_ids.shape == (N,)


def test_edge_index_valid(dataset):
    item = dataset[0]
    assert item.edge_index.shape[0] == 2
    assert item.edge_index.max() < N
    assert item.edge_index.min() >= 0


def test_num_nodes(dataset):
    item = dataset[0]
    assert item.num_nodes == N


@pytest.mark.parametrize("strategy", ["dense", "top_k", "absolute_threshold", "fisher_z_then_threshold"])
def test_fc_strategies(strategy):
    ts = torch.randn(N, 100)
    fc = pearson_correlation(ts)
    graph = build_graph(fc, strategy=strategy, top_k=5, threshold=0.3)
    assert graph.edge_index.shape[0] == 2
    assert graph.num_nodes == N


def test_pearson_correlation_range():
    ts = torch.randn(20, 50)
    fc = pearson_correlation(ts)
    assert fc.shape == (20, 20)
    # Correlation must be in [-1, 1]
    assert fc.abs().max() <= 1.0 + 1e-5


def test_top_k_edges_per_node():
    ts = torch.randn(N, 100)
    fc = pearson_correlation(ts)
    k = 5
    graph = build_graph(fc, strategy="top_k", top_k=k)
    src = graph.edge_index[0]
    counts = torch.bincount(src, minlength=N)
    # Only positive connections survive the clamp, so average degree is in (0, 2k].
    avg_degree = counts.float().mean().item()
    assert 0 < avg_degree <= 2 * k


def test_no_self_loops_by_default():
    ts = torch.randn(N, 50)
    fc = pearson_correlation(ts)
    graph = build_graph(fc, strategy="top_k", top_k=5, self_loops=False)
    src, dst = graph.edge_index
    assert (src == dst).sum() == 0
