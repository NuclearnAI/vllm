#!/usr/bin/env python3
"""
Basic test script to verify Qwen2.5-VLTS integration with vLLM.
This tests model loading and basic functionality.
"""

import sys
import os
import torch

# Add vLLM to path
sys.path.insert(0, '/home/nate/vllm')

def test_model_registration():
    """Test that the model is properly registered in vLLM."""
    print("Testing model registration...")

    try:
        from vllm.model_executor.models.registry import ModelRegistry

        # Check if our model is in the registry
        supported_archs = ModelRegistry.get_supported_archs()

        if "Qwen2_5_VLTSForConditionalGeneration" in supported_archs:
            print("✅ Qwen2_5_VLTSForConditionalGeneration is registered in ModelRegistry")
            return True
        else:
            print("❌ Qwen2_5_VLTSForConditionalGeneration NOT found in ModelRegistry")
            print(f"Available architectures: {sorted(supported_archs)}")
            return False

    except Exception as e:
        print(f"❌ Error testing model registration: {e}")
        return False

def test_model_import():
    """Test that the model can be imported."""
    print("\nTesting model import...")

    try:
        from vllm.model_executor.models.qwen2_5_vlts import (
            Qwen2_5_VLTSForConditionalGeneration,
            Qwen2_5_VLTSConfig,
            Qwen2_5_TimeSeriesPreprocessor
        )
        print("✅ Successfully imported Qwen2.5-VLTS model classes")
        return True

    except Exception as e:
        print(f"❌ Error importing model classes: {e}")
        return False

def test_config_loading():
    """Test config loading from the actual model directory."""
    print("\nTesting config loading...")

    model_path = "/mlshared/nate/wer-multimodal-llm/wer_multimodal_llm/Qwen2.5-VLTS-7B-Base"

    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False

    try:
        from vllm.model_executor.models.qwen2_5_vlts import (
            Qwen2_5_VLTSConfig,
            Qwen2_5_TimeSeriesPreprocessor
        )
        import json

        # Test main config loading
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        config = Qwen2_5_VLTSConfig(**config_data)
        print(f"✅ Successfully loaded main config:")
        print(f"   - Architecture: {config_data.get('architectures', [])}")
        print(f"   - Time-series token ID: {config.time_series_token_id}")
        print(f"   - Time-series config depth: {config.time_series_config.depth}")
        print(f"   - Time-series patch size: {config.time_series_config.patch_size}")

        # Test time-series preprocessor config loading
        ts_preprocessor = Qwen2_5_TimeSeriesPreprocessor.from_pretrained(model_path)
        ts_config = ts_preprocessor.to_dict()
        print(f"   - TS preprocessor config: {ts_config}")

        return True

    except Exception as e:
        print(f"❌ Error loading configs: {e}")
        return False

def test_model_class_loading():
    """Test loading the model class through vLLM registry."""
    print("\nTesting model class loading...")

    try:
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm.config import ModelConfig

        # Create a mock model config
        model_path = "/mlshared/nate/wer-multimodal-llm/wer_multimodal_llm/Qwen2.5-VLTS-7B-Base"

        # This would normally be created by vLLM's model loading pipeline
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            architectures=["Qwen2_5_VLTSForConditionalGeneration"],
            model_config=ModelConfig(
                model=model_path,
                tokenizer=model_path,
                task="generate",
                served_model_name="qwen2.5-vlts-test",
            )
        )

        print(f"✅ Successfully resolved model class: {model_cls}")
        print(f"   - Resolved architecture: {resolved_arch}")

        return True

    except Exception as e:
        print(f"❌ Error loading model class: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Qwen2.5-VLTS Integration with vLLM")
    print("=" * 50)

    tests = [
        test_model_registration,
        test_model_import,
        test_config_loading,
        test_model_class_loading,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = sum(results)
    total = len(results)

    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {test_func.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Qwen2.5-VLTS integration looks good.")
        print("\nNext steps:")
        print("1. Test actual model instantiation with vLLM")
        print("2. Implement full time-series processor integration")
        print("3. Test end-to-end inference")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
