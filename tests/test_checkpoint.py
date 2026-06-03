import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.checkpoint import save_checkpoint, load_checkpoint

def test_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sub", "ckpt")
        state = {"episode": 100, "reward": 215.4, "epsilon": 0.1}
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["episode"] == 100
        assert abs(loaded["reward"] - 215.4) < 0.001
