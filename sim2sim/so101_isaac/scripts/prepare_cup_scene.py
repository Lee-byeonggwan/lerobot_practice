"""Create a separate editable cup scene, preserving the original LiftCube layers.

Cup dimensions not measured by the user are explicitly provisional. This builds
USD geometry/colliders only; it does not establish task success or camera calibration.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'assets/scenes/table_with_wood_cube/scene.usda'
    output = root / 'assets/scenes/table_with_wood_cube_cup/scene.usda'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite edited scene: {output}')
    original = Usd.Stage.Open(str(source))
    inputs = [Path(layer.realPath) for layer in original.GetUsedLayers() if layer.realPath]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, output.parent)]
    stage.SetDefaultPrim(stage.GetPrimAtPath('/world'))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    tabletop = bbox.ComputeWorldBound(original.GetPrimAtPath('/world/counter_right_main_group')).ComputeAlignedRange().GetMax()[2]
    cube = stage.GetPrimAtPath('/world/cube')
    cube.GetAttribute('xformOp:scale').Set(Gf.Vec3d(2/3))
    oldpos = cube.GetAttribute('xformOp:translate').Get()
    cube.GetAttribute('xformOp:translate').Set(Gf.Vec3d(oldpos[0], oldpos[1], tabletop + .01 + .001))
    # Provisional 3 g; actual real block mass has not been measured.
    stage.GetPrimAtPath('/world/cube/cube').GetAttribute('physics:mass').Set(.003)

    # Original work area is forward along world +Y from robot (.35, -.64).
    # World +X is right when standing behind that robot looking into the work area.
    cup = UsdGeom.Xform.Define(stage, '/world/cup')
    cup.AddTranslateOp().Set(Gf.Vec3d(.47, -.42, tabletop + .001))
    cup.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(cup.GetPrim())
    UsdPhysics.MassAPI.Apply(cup.GetPrim()).CreateMassAttr(.005)
    cup.GetPrim().SetCustomDataByKey('practice:provisional', True)

    material = UsdShade.Material.Define(stage, '/world/Looks/PracticePaper')
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Surface'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.86,.83,.76))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.85)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(cup.GetPrim()).Bind(material)
    height, top_r, bottom_r, thickness = .07, .035, .025, .001
    n = 32
    # Individual convex wall segments preserve the hollow opening. Applying one
    # convex hull to the whole cup would incorrectly close its interior.
    faces = [[0,3,2,1], [4,5,6,7], [1,2,6,5], [3,0,4,7], [0,1,5,4], [2,3,7,6]]
    for i in range(n):
        a,b = 2*math.pi*i/n, 2*math.pi*(i+1)/n
        points = []
        for z,r in [(0,bottom_r),(height,top_r)]:
            for angle,radius in [(a,r-thickness),(a,r),(b,r),(b,r-thickness)]:
                points.append(Gf.Vec3f(radius*math.cos(angle),radius*math.sin(angle),z))
        mesh = UsdGeom.Mesh.Define(stage, f'/world/cup/Wall_{i:02d}')
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([v for face in faces for v in face])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    bottom = UsdGeom.Cylinder.Define(stage, '/world/cup/Bottom')
    bottom.CreateAxisAttr('Z')
    bottom.CreateRadiusAttr(bottom_r)
    bottom.CreateHeightAttr(thickness)
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,thickness/2))
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(output))
    bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    cube_size = bbox.ComputeWorldBound(reopened.GetPrimAtPath('/world/cube')).ComputeAlignedRange().GetSize()
    cup_size = bbox.ComputeWorldBound(reopened.GetPrimAtPath('/world/cup')).ComputeAlignedRange().GetSize()
    assert all(abs(v-.02)<1e-6 for v in cube_size), cube_size
    assert all(abs(v-.07)<1e-6 for v in cup_size), cup_size
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report = {
        'scene':str(output), 'source_hashes_unchanged':hashes,
        'measured_by_user':{'cube_edge_m':.02,'cup_height_m':.07,'cup_diameter_m':.07},
        'interpretation':{'cup_diameter':'nominal outer top diameter; inner/outer not confirmed',
                          'right':'standing behind robot looking toward work area'},
        'provisional':{'cup_bottom_outer_diameter_m':.05,'cup_wall_thickness_m':.001,
                       'cup_bottom_thickness_m':.001,'cup_mass_kg':.005,'cube_mass_kg':.003,
                       'cup_initial_world_xyz_m':[.47,-.42,tabletop+.001]},
        'cube_bounds_size_m':list(cube_size),'cup_bounds_size_m':list(cup_size),
        'cup_colliders':'32 individual convex wall segments plus bottom cylinder; no lid',
        'cup_material':'rigid paper-colored approximation; no paper deformation',
        'physics_rollout_verified':False,'render_verified':False,
        'cameras_calibrated':False,'dataset_recording_started':False,
    }
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
