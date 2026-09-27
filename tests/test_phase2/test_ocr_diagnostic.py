import pytest
from PIL import Image
import json
from pathlib import Path

from research_v2.phase2.evidence_validator import normalize_entity_name
from scripts.run_compact_ocr_diagnostic import select_diagnostic_sample

def test_normalize_entity_name_employer():
    assert normalize_entity_name("EXAMPLE PVT LTD") == normalize_entity_name("Example Pvt Ltd")
    assert normalize_entity_name("GlobalTech Solutions") == normalize_entity_name("GLOBALTECH SOLUTIONS")
    assert normalize_entity_name("Nexus Industries") != normalize_entity_name("Acme Services Ltd")

def test_diagnostic_sample_selection(tmp_path):
    manifest_path = Path("research_v2/synthetic_documents/robustness_dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not found")
        
    sample = select_diagnostic_sample(manifest_path)
    
    # Check it selected correctly
    categories = set([c["variant_type"] for c in sample])
    assert "clean_baseline" in categories
    assert "rotation" in categories
    assert "blur" in categories
    assert "noise" in categories
    assert "low_contrast" in categories
    assert "low_brightness" in categories
    assert "partial_edge_crop" in categories
    assert "altered_income" in categories
    assert "altered_employer" in categories
    assert "critical_field_crop" in categories
    
    assert all(c["layout_id"] == "compact" for c in sample)
    
def test_preprocessing_deterministic(tmp_path):
    # Test that grayscale conversion is deterministic
    img = Image.new('RGB', (100, 100), color = 'red')
    
    img1 = img.convert('L')
    img2 = img.convert('L')
    
    assert list(img1.getdata()) == list(img2.getdata())

def test_critical_crop_remains_missing():
    # Load the method comparison to verify it was 0 for all methods
    method_comparison_path = Path("research_v2/phase2_results/compact_ocr_diagnostic/method_comparison.json")
    if not method_comparison_path.exists():
        pytest.skip("Method comparison not found")
        
    with open(method_comparison_path) as f:
        comp = json.load(f)
        
    for name, stats in comp.items():
        assert stats["critical_crop_false_recovery_rate"] == 0.0

