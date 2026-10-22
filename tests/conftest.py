import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import pytest


@pytest.fixture(scope="session")
def gl():
    """ONE standalone OpenGL context for the whole test session (several contexts created/released in one process confuse moderngl)."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context()
    except Exception:
        pytest.skip("no OpenGL context available")
    yield ctx
    ctx.release()
