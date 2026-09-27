import pytest
from pathlib import Path
from PIL import Image
import os
import json
import subprocess

from research_v2.phase2.ocr_adapter import extract_image_text, OCRResult
from research_v2.phase2.evidence_validator import normalize_entity_name

def test_ocr_default_behaviour(tmp_path):
    img = Image.new("RGB", (100, 100), color="white")
    p = tmp_path / "test.png"
    img.save(p)
    
    # Need to mock subprocess.run so tesseract isn't actually run, or run it and it gives empty
    # Since it's a real run, tesseract on empty image gives valid=False but that's fine.
    res = extract_image_text(p)
    assert not res.valid
    
def test_compact_grayscale_psm6(tmp_path, monkeypatch):
    img = Image.new("RGB", (100, 100), color="white")
    p = tmp_path / "test.png"
    img.save(p)
    
    # Mock subprocess.run to check arguments
    run_args = []
    def mock_run(args, **kwargs):
        run_args.append(args)
        # return a fake completed process
        class FakeCompleted:
            returncode = 0
            stdout = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t\n5\t1\t1\t1\t1\t1\t0\t0\t50\t20\t95\tMocked"
        return FakeCompleted()
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    
    res = extract_image_text(p, profile="compact_grayscale_psm6")
    
    assert len(run_args) == 1
    args = run_args[0]
    # Check that it asks for psm 6
    assert "--psm" in args
    assert args[args.index("--psm") + 1] == "6"
    # Check that temp file was created and deleted
    tmp_img_path = args[args.index("stdout") - 1]
    assert tmp_img_path != str(p)
    assert not os.path.exists(tmp_img_path)
    
def test_employer_normalized_accuracy():
    assert normalize_entity_name("GlobalTech Solutions") == normalize_entity_name("GLOBALTECH SOLUTIONS")
    assert normalize_entity_name("Innovatech Corp") != normalize_entity_name("Innovatech Gorp")
