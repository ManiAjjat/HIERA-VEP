import numpy as np
from src.hiera_vep_score import score_frame

def test_exact_additivity(example, assets):
    d=score_frame(example,*assets)
    assert np.isfinite(d.reconstruction_error).all()
    assert d.reconstruction_error.max()<1e-5
    blocks=[c for c in d if c.startswith("contribution_")]
    assert len(blocks)==7
    np.testing.assert_allclose(d[blocks].sum(axis=1),d.hiera_vep_prediction,rtol=0,atol=1e-5)
