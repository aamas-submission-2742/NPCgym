"""Experiment bolt encodings retain their selected DFA IDs and dimensions."""

import hashlib
import json

import pytest

from experiments import merchant_bolts, pacman_bolts

# Snapshots of selected immutable definitions, including minimized Pacman DFAs.
# Include every atom, accepting state, guard and destination, not research outputs.
FINGERPRINTS = {
    "pacman_bolts/vegan": "35cfd91465235b6b64e0e5a869e12cd6436791c995f1767695e17d530d31f954",
    "pacman_bolts/vegetarian": "dda27b71269ee3d4a57ec9ab236c1d5f573c4d81cf3a255a2e44c16a53b89537",
    "pacman_bolts/hungry-vegan": "da9dfb38d87bfa774af73d5d22b181eca1bb793302c0bd4c2859ff5910526e84",
    "pacman_bolts/trapped": "8d5a8af593e3117a24e1f1b204f12d09f04196ce7e746242caa6e0db9cf1b6bd",
    "pacman_bolts/vegan-conflict": "2e306fd691cdd070a16cf61074131fc7f722db434014a1ebe3eddb1dc86997eb",
    "pacman_bolts/hungry-vegan-penalty": "a924413212b059833e66bcbc153f2ea8131cab749b21f99adb2c110a234eae05",
    "merchant_bolts/env-friendly": "a026764caa49e31bc1df8534697bfc5b37972e4e1642e715b1ed2900587777df",
    "merchant_bolts/delivery-pacifist": "e40682abd39df204160811851e21565c31bf995798fcf5b824b1b251680cca02",
}


@pytest.mark.parametrize("key,expected", FINGERPRINTS.items())
def test_selected_experiment_bolt_encodings(key, expected):
    module_name, norm = key.split("/")
    module = {"merchant_bolts": merchant_bolts, "pacman_bolts": pacman_bolts}[module_name]
    bolts = module.make_bolts(norm)
    data = {
        name: {
            "atoms": spec.definition.atoms,
            "initial": spec.definition.initial_state,
            "finals": sorted(spec.definition.final_states),
            "transitions": dict(spec.definition.transitions),
        }
        for name, spec in bolts.items()
    }
    assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == expected
