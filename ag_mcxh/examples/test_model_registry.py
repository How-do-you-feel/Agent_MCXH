import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
ag_mcxh_root = os.path.join(project_root, 'ag_mcxh')

print(f"Current directory: {current_dir}")
print(f"Project root: {project_root}")
print(f"AG_Mcxh root: {ag_mcxh_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if ag_mcxh_root not in sys.path:
    sys.path.insert(0, ag_mcxh_root)

def test_model_registry():
    try:
        from ag_mcxh.models.registry import list_models, load_model, get_model_cls
        print("✓ Successfully imported model registry")
        
        models = list_models()
        print(f"Registered models: {models}")
        
        if models:
            first_model = models[0]
            print(f"Testing model: {first_model}")
            model_cls = get_model_cls(first_model)
            print(f"✓ Got model class: {model_cls}")
            
            try:
                model_instance = load_model(first_model)
                print(f"✓ Loaded model instance: {type(model_instance)}")
            except Exception as e:
                print(f"⚠ Could not load model instance: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test model registry: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tools_with_models():
    try:
        from ag_mcxh.apis import list_tools, load_tool
        print("✓ Successfully imported tools")
        
        tools = list_tools()
        print(f"Available tools: {tools}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test tools: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Model Registry System")
    print("=" * 40)
    
    success1 = test_model_registry()
    print()
    success2 = test_tools_with_models()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

if __name__ == "__main__":
    main()