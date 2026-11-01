"""Run USD scene authoring inside an initialized Isaac Sim application."""
from isaaclab.app import AppLauncher
import argparse

p = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
launcher = AppLauncher(a)
try:
    import prepare_wood_cube, prepare_cup_scene, prepare_real_style_layout, prepare_collection_scene
    import sys
    sys.argv = [sys.argv[0]]  # The authoring scripts have their own parsers.
    for module in (prepare_wood_cube, prepare_cup_scene, prepare_real_style_layout, prepare_collection_scene):
        module.main()
finally:
    launcher.app.close()
