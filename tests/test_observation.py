"""Root adapter shares the trainer's observation and dimensions."""
from common.config import nominal_case,ELBOW_TO_WRIST,WRIST_TO_PAN
from toss.domain import observe

def test_shared_observation_contract():
    c=nominal_case()
    assert observe(c).shape==(23,)
    assert ELBOW_TO_WRIST==.24 and WRIST_TO_PAN==.1
