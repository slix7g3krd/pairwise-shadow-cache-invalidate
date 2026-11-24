import pytest
import graphics.shadow_cache_invalidate as mod

CASES = [
    ("dynamic_channel", (1,)),
    ("light_invalidate", (1,)),
    ("page_fence", (1,)),
    ("cascade_round", (1,)),
    ("dirty_aggregate", (1,)),
    ("rebuild", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("cache_page", (1,)),
    ("metrics", (1,)),
    ("shadow_render", (1,)),
    ("cascade_select", (1,)),
    ("invalidate", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
