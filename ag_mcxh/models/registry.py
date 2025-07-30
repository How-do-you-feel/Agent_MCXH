from typing import Dict, Type, Any, Callable
import importlib

# Model registry, mapping string identifiers to model classes or loader functions
MODELS_REGISTRY: Dict[str, Callable[[], Any]] = {}

def register_model(name: str):
    """Model registration decorator"""
    def decorator(cls: Type[Any]):
        # Store a lazy loading function instead of the class directly
        def loader():
            return cls
        MODELS_REGISTRY[name] = loader
        return cls
    return decorator

def register_model_loader(name: str, loader: Callable[[], Any]):
    """Register model loader function"""
    MODELS_REGISTRY[name] = loader

def get_model_cls(name: str) -> Type[Any]:
    """Get model class (lazy loading)"""
    if name not in MODELS_REGISTRY:
        raise ValueError(f"Model {name} is not registered")
    
    # Call the loader function to get the model class
    loader = MODELS_REGISTRY[name]
    return loader()

def list_models() -> list:
    """List all registered models"""
    return list(MODELS_REGISTRY.keys())

def load_model(name: str, *args, **kwargs) -> Any:
    """Load model instance"""
    model_cls = get_model_cls(name)
    return model_cls(*args, **kwargs)

# Automatically register common model lazy loading functions
def _register_common_models():
    """Register common models with lazy loading functions to avoid import dependencies"""
    
    # YOLO model lazy loading
    def _load_yolo_model():
        try:
            from ultralytics import YOLO
            return YOLO
        except ImportError:
            class MockYOLO:
                def __init__(self, *args, **kwargs):
                    pass
                def to(self, device):
                    pass
                def __call__(self, *args, **kwargs):
                    return []
            return MockYOLO
    
    register_model_loader("YOLO", _load_yolo_model)
    
    # SAM model lazy loading
    def _load_sam_model():
        try:
            from segment_anything import sam_model_registry
            return sam_model_registry
        except ImportError:
            class MockSAMRegistry:
                def __getitem__(self, key):
                    class MockSAM:
                        def __init__(self, *args, **kwargs):
                            pass
                        def to(self, device):
                            pass
                    return MockSAM
            return MockSAMRegistry()
    
    register_model_loader("SAM", _load_sam_model)

# Initialize registry
_register_common_models()