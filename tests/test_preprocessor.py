"""
tests/test_preprocessor.py
Testes unitários do módulo preprocessor.py.
"""
import numpy as np
import pytest
from preprocessing.preprocessor import Preprocessor, PreprocessConfig

def make_frame(h=480, w=640, dtype=np.uint8):
    return np.random.randint(0, 255, (h, w, 3), dtype=dtype)

class TestPreprocessorOutput:
    def test_output_shape_letterbox(self):
        pp  = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame())
        assert res.frame.shape == (416, 416, 3)

    def test_output_dtype_uint8(self):
        pp  = Preprocessor(PreprocessConfig(normalize=False))
        res = pp.process(make_frame())
        assert res.frame.dtype == np.uint8

    def test_output_dtype_float32_when_normalized(self):
        pp  = Preprocessor(PreprocessConfig(normalize=True))
        res = pp.process(make_frame())
        assert res.frame.dtype == np.float32
        assert res.frame.max() <= 1.0

    def test_scale_and_padding_set(self):
        pp  = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=True))
        res = pp.process(make_frame(h=480, w=640))
        assert res.scale > 0
        assert res.orig_size == (480, 640)

    def test_letterbox_padding_symmetric(self):
        pp  = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=True))
        res = pp.process(make_frame(h=416, w=416))
        assert res.pad_w == 0
        assert res.pad_h == 0

class TestBboxAdjustment:
    def test_adjust_removes_letterbox_offset(self):
        pp  = Preprocessor(PreprocessConfig(infer_size=416))
        res = pp.process(make_frame(h=480, w=640))
        boxes_lb = np.array([[10, 50, 100, 200]], dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_lb, res)
        if res.pad_h > 0:
            assert boxes_orig[0, 1] < boxes_lb[0, 1]

class TestPreprocessorConfigs:
    def test_config_low_light_applies_clahe(self):
        from preprocessing.preprocessor import CONFIG_LOW_LIGHT
        pp  = Preprocessor(CONFIG_LOW_LIGHT)
        res = pp.process(make_frame())
        assert res.frame.shape[2] == 3

    def test_config_default_no_filter(self):
        from preprocessing.preprocessor import CONFIG_DEFAULT
        pp = Preprocessor(CONFIG_DEFAULT)
        assert not pp.cfg.gaussian_blur
        assert not pp.cfg.median_blur
        assert not pp.cfg.clahe

class TestNonUniformScale:
    def test_adjust_boxes_without_letterbox_uses_separate_axis_scales(self):
        pp  = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=False))
        res = pp.process(make_frame(h=480, w=640))
        assert res.scale_x != res.scale_y
        boxes_resized = np.array([[0, 0, 416, 416]], dtype=float)
        boxes_orig = pp.adjust_boxes(boxes_resized, res)
        assert abs(boxes_orig[0, 2] - 640) < 1e-6
        assert abs(boxes_orig[0, 3] - 480) < 1e-6
