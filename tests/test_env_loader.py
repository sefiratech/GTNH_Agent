from env.loader import load_environment, EnvProfile

def test_load_environment_returns_envprofile():
    profile = load_environment()
    assert isinstance(profile, EnvProfile)

