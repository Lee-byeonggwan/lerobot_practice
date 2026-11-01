"""Create a visual-only USD overlay; never modify the official LiftCube assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physics_snapshot(stage):
    return {
        str(prim.GetPath()): {
            attr.GetName(): str(attr.Get())
            for attr in prim.GetAttributes()
            if attr.GetName().startswith(("physics:", "physx", "xformOp", "points", "extent"))
        }
        for prim in stage.Traverse()
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "assets/scenes/table_with_cube/scene.usd")
    parser.add_argument("--output", type=Path, default=root / "assets/scenes/table_with_wood_cube/scene.usda")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output == source:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    original_hash = digest(source)
    original = Usd.Stage.Open(str(source))
    visual_path = "/world/cube/cube/Visuals/cube"
    if not original.GetPrimAtPath(visual_path).IsA(UsdGeom.Mesh):
        raise ValueError(f"Official cube visual mesh missing: {visual_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, output.parent)]
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(original))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(original))
    material = UsdShade.Material.Define(stage, "/world/Looks/PracticeWoodColor")
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    # Linear RGB: warm light-brown appearance, not a wood-grain texture or physics change.
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.24, 0.10))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(visual_path)).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(output))
    before, after = physics_snapshot(original), physics_snapshot(reopened)
    assert all(after[path] == values for path, values in before.items()), "Physics or geometry changed"
    assert original_hash == digest(source), "Source asset changed"
    bound, _ = UsdShade.MaterialBindingAPI(reopened.GetPrimAtPath(visual_path)).ComputeBoundMaterial()
    assert bound.GetPath() == material.GetPath(), "Wood material binding did not resolve"
    report = {
        "source": str(source), "source_sha256": original_hash,
        "output": str(output), "visual_mesh": visual_path,
        "material": str(bound.GetPath()), "physics_geometry_unchanged": True,
        "render_verified": False,
    }
    output.with_suffix(".validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
