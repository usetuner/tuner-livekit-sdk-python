import subprocess
import sys


def test_telephony_import_does_not_load_livekit_agents():
    code = "import sys, tuner.telephony; assert 'livekit.agents' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tuner_plugin_still_importable():
    from tuner import TunerPlugin

    assert TunerPlugin.__name__ == "TunerPlugin"
