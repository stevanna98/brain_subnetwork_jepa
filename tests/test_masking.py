"""Tests for subnetwork masking correctness."""

import pytest
import torch

from brain_jepa.data.atlas import load_atlas
from brain_jepa.data.dataset import SyntheticBrainDataset
from brain_jepa.masking import SubnetworkMaskCollator

ATLAS_CSV = "data/atlas/glasser379_to_rsn12.csv"
NUM_RSNS = 12
NUM_REGIONS = 379


@pytest.fixture(scope="module")
def atlas():
    return load_atlas(ATLAS_CSV)


@pytest.fixture(scope="module")
def synthetic_dataset(atlas):
    return SyntheticBrainDataset(atlas=atlas, num_subjects=8, feature_dim=32, seed=0)


@pytest.fixture(scope="module")
def collated(atlas, synthetic_dataset):
    collator = SubnetworkMaskCollator(num_rsns=NUM_RSNS, num_targets=1)
    batch = [synthetic_dataset[i] for i in range(4)]
    return collator(batch)


def test_context_target_disjoint(collated):
    """Context and target RSN sets must be disjoint for every sample."""
    _, masks = collated
    B = masks.context_rsn_ids.shape[0]
    for b in range(B):
        ctx = set(masks.context_rsn_ids[b].tolist())
        tgt = set(masks.target_rsn_ids[b].tolist())
        assert ctx.isdisjoint(tgt), f"Sample {b}: overlap between context and target RSNs"


def test_context_target_partition(collated):
    """Context ∪ target = all RSNs; |context| + |target| = K."""
    _, masks = collated
    B = masks.context_rsn_ids.shape[0]
    for b in range(B):
        ctx = set(masks.context_rsn_ids[b].tolist())
        tgt = set(masks.target_rsn_ids[b].tolist())
        assert ctx | tgt == set(range(NUM_RSNS)), f"Sample {b}: union ≠ all RSNs"
        assert len(ctx) + len(tgt) == NUM_RSNS


def test_node_masks_disjoint(collated):
    """Context and target node masks must not overlap."""
    _, masks = collated
    B = len(masks.context_node_masks)
    for b in range(B):
        overlap = masks.context_node_masks[b] & masks.target_node_masks[b]
        assert not overlap.any(), f"Sample {b}: context and target node masks overlap"


def test_node_masks_cover_all_nodes(collated):
    """Context ∪ target node masks cover all 379 nodes."""
    _, masks = collated
    B = len(masks.context_node_masks)
    for b in range(B):
        union = masks.context_node_masks[b] | masks.target_node_masks[b]
        assert union.all(), f"Sample {b}: some nodes not assigned to context or target"


@pytest.mark.parametrize("num_targets", [1, 2, 3])
def test_variable_num_targets(atlas, synthetic_dataset, num_targets):
    """Masking works for different values of M."""
    collator = SubnetworkMaskCollator(num_rsns=NUM_RSNS, num_targets=num_targets)
    batch = [synthetic_dataset[i] for i in range(4)]
    _, masks = collator(batch)
    assert masks.target_rsn_ids.shape == (4, num_targets)
    assert masks.context_rsn_ids.shape == (4, NUM_RSNS - num_targets)


def test_extra_target_ratio(atlas, synthetic_dataset):
    """Extra random nodes are masked on top of the target RSN; the node masks
    still partition all nodes and target counts vary across samples."""
    ratio = 0.15
    collator = SubnetworkMaskCollator(
        num_rsns=NUM_RSNS, num_targets=1, extra_target_ratio=ratio
    )
    batch = [synthetic_dataset[i] for i in range(4)]
    _, masks = collator(batch)
    for b in range(4):
        tgt_mask = masks.target_node_masks[b]
        ctx_mask = masks.context_node_masks[b]
        rsn_ids = batch[b].rsn_ids
        # Still disjoint and covering
        assert not (tgt_mask & ctx_mask).any()
        assert (tgt_mask | ctx_mask).all()
        # All target-RSN nodes are masked...
        rsn_tgt = torch.isin(rsn_ids, masks.target_rsn_ids[b])
        assert tgt_mask[rsn_tgt].all()
        # ...plus approximately ratio * remaining extra nodes
        n_extra = int(tgt_mask.sum()) - int(rsn_tgt.sum())
        expected = round(ratio * int((~rsn_tgt).sum()))
        assert n_extra == expected


def test_extra_targets_differ_across_samples(atlas, synthetic_dataset):
    """With extra random targets, two samples masking the same RSN should
    still have different target node sets (task variety)."""
    collator = SubnetworkMaskCollator(
        num_rsns=NUM_RSNS, num_targets=1, extra_target_ratio=0.15
    )
    data = synthetic_dataset[0]
    torch.manual_seed(0)
    _, masks_a = collator([data, data])
    # Same subject twice — extras are sampled independently per subject
    a, b = masks_a.target_node_masks
    if masks_a.target_rsn_ids[0].item() == masks_a.target_rsn_ids[1].item():
        assert not torch.equal(a, b)


def test_atlas_region_count(atlas):
    assert atlas.num_regions == NUM_REGIONS


def test_atlas_rsn_count(atlas):
    assert atlas.num_rsns == NUM_RSNS


def test_atlas_rsn_ids_valid(atlas):
    assert atlas.rsn_ids.min() == 0
    assert atlas.rsn_ids.max() == NUM_RSNS - 1


def test_rsn_mask_tensor_shape(atlas):
    mask = atlas.rsn_mask_tensor()
    assert mask.shape == (NUM_RSNS, NUM_REGIONS)
    assert mask.dtype == torch.bool
    # Each region belongs to exactly one RSN
    assert mask.sum(dim=0).eq(1).all()
