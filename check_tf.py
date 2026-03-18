import sys
print("Python:", sys.version)
print("Executable:", sys.executable)
try:
    import tensorflow as tf
    print("TF version:", getattr(tf, "__version__", "unknown"))
    try:
        print("TF file:", getattr(tf, "__file__", "unknown"))
    except Exception:
        pass
    try:
        # Ensure tensorflow.keras is available
        from tensorflow.keras.models import load_model  # noqa: F401
        print("tensorflow.keras import: OK")
    except Exception as e:
        print("tensorflow.keras import: FAIL:", repr(e))
except Exception as e:
    print("tensorflow import: FAIL:", repr(e))

