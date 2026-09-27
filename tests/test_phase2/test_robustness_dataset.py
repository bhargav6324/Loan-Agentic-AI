import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageStat

from scripts.generate_robustness_dataset import generate_dataset, validate_dataset

def test_generate_and_validate_dataset(tmp_path):
    output_dir = tmp_path / "images"
    manifest_path = tmp_path / "manifest.json"
    inspection_dir = tmp_path / "inspection"
    
    manifest1 = generate_dataset(
        output_dir,
        manifest_path,
        seed=42,
        size=60,
        inspection_dir=inspection_dir,
    )
    assert len(manifest1) == 60
    
    # Test validation
    assert validate_dataset(manifest_path) is True
    
    # Test reproducibility
    manifest_path2 = tmp_path / "manifest2.json"
    manifest2 = generate_dataset(
        output_dir,
        manifest_path2,
        seed=42,
        size=60,
        inspection_dir=inspection_dir,
    )
    
    for r1, r2 in zip(manifest1, manifest2):
        assert r1["variant_id"] == r2["variant_id"]
        assert r1["image_path"] == r2["image_path"]
        assert r1["expected_fault_code"] == r2["expected_fault_code"]


def test_dataset_layouts_severity_regions_and_isolation(tmp_path):
    output_dir = tmp_path / "images"
    manifest_path = tmp_path / "manifest.json"
    inspection_dir = tmp_path / "inspection"
    manifest = generate_dataset(
        output_dir,
        manifest_path,
        seed=42,
        size=60,
        inspection_dir=inspection_dir,
    )

    layout_counts = Counter(row["layout_id"] for row in manifest)
    assert set(layout_counts) == {"standard", "compact", "tabular"}
    assert max(layout_counts.values()) - min(layout_counts.values()) <= 10

    allowed_severities = {"none", "low", "medium", "high"}
    assert {row["severity"] for row in manifest} <= allowed_severities
    assert all(
        row["severity"] == "none"
        for row in manifest
        if row["variant_type"] == "clean_baseline"
    )
    assert all(
        row["severity"] == "high"
        for row in manifest
        if row["variant_type"] in {
            "critical_field_crop",
            "altered_income",
            "altered_employer",
        }
    )
    assert all(
        row["rotation_degrees"] is not None
        for row in manifest
        if row["variant_type"] == "rotation"
    )
    assert all(
        row["rotation_degrees"] is None
        for row in manifest
        if row["variant_type"] != "rotation"
    )

    regions = {
        row["layout_id"]: tuple(row["critical_field_region"])
        for row in manifest
    }
    assert len(set(regions.values())) == 3

    for layout_id in layout_counts:
        clean = next(
            row
            for row in manifest
            if row["layout_id"] == layout_id
            and row["variant_type"] == "clean_baseline"
        )
        masked = next(
            row
            for row in manifest
            if row["case_id"] == clean["case_id"]
            and row["variant_type"] == "critical_field_crop"
        )
        region = tuple(masked["critical_field_region"])
        clean_image = Image.open(clean["image_path"]).convert("L")
        masked_image = Image.open(masked["image_path"]).convert("L")
        left, top, right, bottom = region
        assert 0 <= left < right <= clean_image.width
        assert 0 <= top < bottom <= clean_image.height
        assert ImageStat.Stat(clean_image.crop(region)).extrema[0][0] < 200
        assert ImageStat.Stat(masked_image.crop(region)).extrema[0][0] >= 250

    assert len({row["variant_id"] for row in manifest}) == len(manifest)
    assert len({row["image_path"] for row in manifest}) == len(manifest)
    assert len(list(output_dir.glob("case-*.png"))) == len(manifest)
    assert len(list(inspection_dir.glob("*.png"))) == 15
    index = json.loads((inspection_dir / "index.json").read_text())
    assert len(index) == 15
    assert all(str(tmp_path) in row["source_image"] for row in index)


def test_dataset_size_must_be_positive_multiple_of_ten(tmp_path):
    with pytest.raises(ValueError, match="positive multiple of 10"):
        generate_dataset(
            tmp_path / "images",
            tmp_path / "manifest.json",
            size=51,
            inspection_dir=tmp_path / "inspection",
        )
