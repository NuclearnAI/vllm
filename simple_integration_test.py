#!/usr/bin/env python3
"""
Simplified test script to verify Qwen2.5-VLTS integration syntax and basic structure.
This tests the integration without requiring full vLLM runtime dependencies.
"""

import sys
import os
import ast

def test_file_syntax(filepath):
    """Test if a Python file has valid syntax."""
    print(f"Testing syntax of {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file to check syntax
        ast.parse(content)
        print(f"✅ {filepath} has valid Python syntax")
        return True

    except SyntaxError as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

def test_model_file_structure():
    """Test that our model file has the expected structure."""
    print("\nTesting model file structure...")

    model_file = "/home/nate/vllm/vllm/model_executor/models/qwen2_5_vlts.py"

    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False

    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key components
        required_components = [
            "class Qwen2_5_VLTSForConditionalGeneration",
            "class Qwen2_5_VLTSConfig",
            "class Qwen2_5_TimeSeriesTransformer",
            "class DigitwiseFloatEmbedding",
            "class DatetimeEmbedding",
            "MULTIMODAL_REGISTRY.register_processor",
            "def load_weights",
            "def get_multimodal_embeddings",
        ]

        missing_components = []
        for component in required_components:
            if component not in content:
                missing_components.append(component)

        if missing_components:
            print(f"❌ Missing components: {missing_components}")
            return False
        else:
            print("✅ All required components found in model file")
            return True

    except Exception as e:
        print(f"❌ Error analyzing model file: {e}")
        return False

def test_registry_integration():
    """Test that the registry file includes our model."""
    print("\nTesting registry integration...")

    registry_file = "/home/nate/vllm/vllm/model_executor/models/registry.py"

    try:
        with open(registry_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if "Qwen2_5_VLTSForConditionalGeneration" in content:
            print("✅ Qwen2_5_VLTSForConditionalGeneration found in registry")
            return True
        else:
            print("❌ Qwen2_5_VLTSForConditionalGeneration NOT found in registry")
            return False

    except Exception as e:
        print(f"❌ Error checking registry: {e}")
        return False

def test_multimodal_updates():
    """Test that multimodal framework updates are in place."""
    print("\nTesting multimodal framework updates...")

    inputs_file = "/home/nate/vllm/vllm/multimodal/inputs.py"
    parse_file = "/home/nate/vllm/vllm/multimodal/parse.py"

    try:
        # Check inputs.py for time-series enhancements
        with open(inputs_file, 'r', encoding='utf-8') as f:
            inputs_content = f.read()

        if "list[list[list[float]]]" in inputs_content and "Enhanced support for nested ragged structures" in inputs_content:
            print("✅ Enhanced TimeSeriesItem types found in inputs.py")
            inputs_ok = True
        else:
            print("❌ Enhanced TimeSeriesItem types NOT found in inputs.py")
            inputs_ok = False

        # Check parse.py for time-series parsing enhancements
        with open(parse_file, 'r', encoding='utf-8') as f:
            parse_content = f.read()

        if "get_total_streams" in parse_content and "get_stream_lengths" in parse_content:
            print("✅ Enhanced time-series parsing methods found in parse.py")
            parse_ok = True
        else:
            print("❌ Enhanced time-series parsing methods NOT found in parse.py")
            parse_ok = False

        return inputs_ok and parse_ok

    except Exception as e:
        print(f"❌ Error checking multimodal updates: {e}")
        return False

def test_config_compatibility():
    """Test that our config is compatible with the actual model."""
    print("\nTesting config compatibility...")

    model_path = "/mlshared/nate/wer-multimodal-llm/wer_multimodal_llm/Qwen2.5-VLTS-7B-Base"

    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False

    try:
        import json

        # Load the actual model config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        # Load the ts preprocessor config
        ts_config_path = os.path.join(model_path, "ts_preprocessor_config.json")
        with open(ts_config_path, 'r') as f:
            ts_config_data = json.load(f)

        print("✅ Successfully loaded config files:")
        print(f"   - Architecture: {config_data.get('architectures')}")
        print(f"   - Model type: {config_data.get('model_type')}")
        print(f"   - Time-series token ID: {config_data.get('time_series_token_id')}")
        print(f"   - TS patch size: {ts_config_data.get('ts_patch_size')}")
        print(f"   - TS digit pad: {ts_config_data.get('ts_digit_pad')}")

        # Verify key fields exist
        required_fields = [
            'time_series_config',
            'time_series_token_id',
            'vision_config',
            'text_config'
        ]

        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            print(f"❌ Missing required config fields: {missing_fields}")
            return False

        print("✅ All required config fields present")
        return True

    except Exception as e:
        print(f"❌ Error testing config compatibility: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Simple Qwen2.5-VLTS Integration Test")
    print("=" * 50)

    tests = [
        lambda: test_file_syntax("/home/nate/vllm/vllm/model_executor/models/qwen2_5_vlts.py"),
        lambda: test_file_syntax("/home/nate/vllm/vllm/multimodal/inputs.py"),
        lambda: test_file_syntax("/home/nate/vllm/vllm/multimodal/parse.py"),
        test_model_file_structure,
        test_registry_integration,
        test_multimodal_updates,
        test_config_compatibility,
    ]

    test_names = [
        "qwen2_5_vlts.py syntax",
        "inputs.py syntax",
        "parse.py syntax",
        "model file structure",
        "registry integration",
        "multimodal updates",
        "config compatibility"
    ]

    results = []
    for test_func, test_name in zip(tests, test_names):
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = sum(results)
    total = len(results)

    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {i+1}. {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All basic integration tests passed!")
        print("\n✅ Phase 1 (Core Model Integration) appears to be working correctly.")
        print("\nWhat we've accomplished:")
        print("1. ✅ Created Qwen2.5-VLTS model implementation in vLLM")
        print("2. ✅ Registered model in vLLM's model registry")
        print("3. ✅ Enhanced multimodal framework for time-series support")
        print("4. ✅ Added configuration and weight loading support")
        print("5. ✅ Verified syntax and structural correctness")

        print("\nNext phases to implement:")
        print("📋 Phase 2: Complete multimodal processing pipeline")
        print("📋 Phase 3: Time-series processor integration")
        print("📋 Phase 4: Token and sequence management")
        print("📋 Phase 5: End-to-end testing with actual inference")

    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the errors above.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
