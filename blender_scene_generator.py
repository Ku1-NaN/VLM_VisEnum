"""
blender_scene_generator.py

Blender background script for procedural scene generation.
This script reads a JSON configuration file, populates a 3D scene with 
primitives, calculates overlap/collision to ensure valid placements, 
generates segmentation masks, and renders the final scene using Cycles.

Author: Kuinan
Usage:
    blender --background --python blender_scene_generator.py -- /path/to/config.json
"""

import sys
import os
import json
import random
import math
import mathutils
import numpy as np
import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
import traceback
from math import radians

# ------------------ Helper: read config path from blender args ------------------
def get_cli_config_path():
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--")
        user_args = argv[idx+1:]
    else:
        user_args = []
    if len(user_args) < 1:
        raise RuntimeError("Expecting one argument: path to JSON config file")
    return user_args[0]

cfg_path = get_cli_config_path()
with open(cfg_path, "r") as f:
    cfg = json.load(f)

# ------------------ Read parameters from config ------------------
object_specs = cfg["object_specs"]
target_count = int(cfg["target_count"])
repetition_index = int(cfg.get("repetition_index", 0))
seed = int(cfg.get("seed", 0))
attempts_per_image = int(cfg.get("attempts_per_image", 5))
max_attempts_per_shape = int(cfg.get("max_attempts_per_shape", 500))
output_image = cfg["output_image"]
output_config = cfg["output_config"]
z_offset = float(cfg.get("z_offset", 0.5))
min_distance_world = float(cfg.get("min_distance_world", 0.2))
min_size_fraction = float(cfg.get("min_size_fraction", 0.06))
max_size_fraction = float(cfg.get("max_size_fraction", 0.16))
allowed_overlap_fraction = float(cfg.get("allowed_overlap_fraction", 0.10))
image_resolution = cfg.get("image_resolution", [1080,1080])
job_name = cfg.get("job_name", "default_job")
mask_dir = cfg.get("mask_dir", bpy.app.tempdir)
job_folder = os.path.join(mask_dir, job_name)
os.makedirs(job_folder, exist_ok=True)

job_temp_dir = cfg.get("temp_dir")
if job_temp_dir:
    os.makedirs(job_temp_dir, exist_ok=True)
else:
    job_temp_dir = bpy.app.tempdir

random.seed(seed + repetition_index)

# ------------------ Scene constants ------------------
bottom_width = cfg.get("bottom_width", 20)
top_width = cfg.get("top_width", 10)
height = cfg.get("height", 20)
edge_margin = cfg.get("edge_margin", 0.7)
background_width = bottom_width + 6
background_height = height + 6
background_z = -0.01
image_overlap_margin = 0.05

# ------------------ Utility functions ------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

def add_shape_primitive(shape_type, location):
    if shape_type == "cube":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    elif shape_type == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location, segments=64, ring_count=32)
    elif shape_type == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1.0, location=location)
    elif shape_type == "cone":
        bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1.0, location=location)
    elif shape_type == "torus":
        bpy.ops.mesh.primitive_torus_add(location=location, major_radius=0.5, minor_radius=0.2)
    elif shape_type == "capsule":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(location[0], location[1], location[2] + 0.3))
        sph_top = bpy.context.active_object
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.25, location=(location[0], location[1], location[2] - 0.3))
        sph_bot = bpy.context.active_object
        bpy.ops.mesh.primitive_cylinder_add(radius=0.25, depth=0.6, location=location)
        cyl = bpy.context.active_object
        bpy.ops.object.select_all(action='DESELECT')
        sph_top.select_set(True)
        sph_bot.select_set(True)
        cyl.select_set(True)
        bpy.context.view_layer.objects.active = cyl
        bpy.ops.object.join()
    elif shape_type == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location, segments=48, ring_count=24)
        ell = bpy.context.active_object
        mesh = ell.data
        for v in mesh.vertices:
            v.co.x *= 1.0
            v.co.y *= 0.8
            v.co.z *= 0.6
        mesh.update()
    elif shape_type == "pyramid":
        mesh = bpy.data.meshes.new("PyramidMesh")
        obj_pyr = bpy.data.objects.new("Pyramid", mesh)
        bpy.context.collection.objects.link(obj_pyr)
        bm = bmesh.new()
        half = 0.5
        v1 = bm.verts.new((-half, -half, -0.5))
        v2 = bm.verts.new((half, -half, -0.5))
        v3 = bm.verts.new((half, half, -0.5))
        v4 = bm.verts.new((-half, half, -0.5))
        top = bm.verts.new((0, 0, 0.6))
        bm.faces.new([v1, v2, v3, v4])
        bm.faces.new([v1, v2, top])
        bm.faces.new([v2, v3, top])
        bm.faces.new([v3, v4, top])
        bm.faces.new([v4, v1, top])
        bm.to_mesh(mesh)
        bm.free()
        obj_pyr.location = location
        bpy.context.view_layer.update()
        return obj_pyr
    else:
        return None

    obj = bpy.context.active_object
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    return obj

def create_material_for_color_rgba(rgba, name_hint="ObjMat"):
    mat = bpy.data.materials.new(name=f"{name_hint}_{random.randint(0,9999)}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs['Base Color'].default_value = (*rgba[:3], 1.0)
    bsdf.inputs['Roughness'].default_value = 0.66
    bsdf.inputs['Metallic'].default_value = 0.05
    return mat

def point_in_trapezoid(x, y):
    if y < edge_margin or y > height - edge_margin:
        return False
    width_at_y = bottom_width + ((top_width - bottom_width) * (y / height))
    half_width = width_at_y / 2
    inner_half_width = half_width - edge_margin
    if inner_half_width < 0:
        return False
    return -inner_half_width <= x <= inner_half_width

def get_world_bbox(obj):
    bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]
    world_corners = [obj.matrix_world @ corner for corner in bbox_corners]
    return world_corners

def project_bbox_to_image(obj, cam, scene):
    corners = get_world_bbox(obj)
    coords2d = [world_to_camera_view(scene, cam, c) for c in corners]
    xs = [max(0.0, min(1.0, c.x)) for c in coords2d]
    ys = [max(0.0, min(1.0, c.y)) for c in coords2d]
    return (min(xs), min(ys), max(xs), max(ys)), coords2d

def normalized_bbox_area(b):
    xmin, ymin, xmax, ymax = b
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return w * h

def bbox_intersection_area(b1, b2):
    x1min, y1min, x1max, y1max = b1
    x2min, y2min, x2max, y2max = b2
    ixmin = max(x1min, x2min)
    iymin = max(y1min, y2min)
    ixmax = min(x1max, x2max)
    iymax = min(y1max, y2max)
    if ixmax <= ixmin or iymax <= iymin:
        return 0.0
    return (ixmax - ixmin) * (iymax - iymin)

def bbox_overlap_fraction(b1, b2):
    a1 = normalized_bbox_area(b1)
    a2 = normalized_bbox_area(b2)
    if a1 <= 0 or a2 <= 0:
        return 1.0
    inter = bbox_intersection_area(b1, b2)
    return inter / min(a1, a2)

def world_xy_footprint_radius(obj):
    bbox_world = get_world_bbox(obj)
    xs = [v.x for v in bbox_world]
    ys = [v.y for v in bbox_world]
    w = max(xs) - min(xs)
    d = max(ys) - min(ys)
    radius = math.hypot(w, d) / 2.0
    return radius

def get_projected_height_pixels(obj, camera, scene):
    bbox, _ = project_bbox_to_image(obj, camera, scene)
    return (bbox[3] - bbox[1]) * scene.render.resolution_y

# ------------------ Scene setup functions ------------------
def setup_scene_and_camera():
    scene = bpy.context.scene
    cam_x = 0
    cam_y = height + 6
    cam_z = 14.5
    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    camera = bpy.context.active_object
    scene.camera = camera
    center = mathutils.Vector((0, height / 2, 0))
    direction = center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    euler = rot_quat.to_euler()
    euler.x += math.radians(-5.5)
    camera.rotation_euler = euler
    cam_data = camera.data
    cam_data.lens = 35.0
    cam_data.sensor_fit = 'VERTICAL'
    cam_data.sensor_height = 24.0
    scene.render.resolution_x = image_resolution[0]
    scene.render.resolution_y = image_resolution[1]
    return scene, camera

def setup_background_and_trapezoid():
    bg_mesh = bpy.data.meshes.new("Background")
    bg_obj = bpy.data.objects.new("Background", bg_mesh)
    bpy.context.collection.objects.link(bg_obj)
    bm_bg = bmesh.new()

    bg_verts = [
        (-background_width / 2, -3, background_z),
        (background_width / 2, -3, background_z),
        (background_width / 2, height + 3, background_z),
        (-background_width / 2, height + 3, background_z),
    ]
    bm_bg_verts = [bm_bg.verts.new(co) for co in bg_verts]
    bm_bg.faces.new(bm_bg_verts)
    bm_bg.to_mesh(bg_mesh)
    bm_bg.free()

    mat_bg = bpy.data.materials.new(name="BackgroundChecker")
    mat_bg.use_nodes = True
    nodes = mat_bg.node_tree.nodes
    links = mat_bg.node_tree.links

    for n in nodes:
        nodes.remove(n)

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    checker = nodes.new(type="ShaderNodeTexChecker")
    color1 = nodes.new(type="ShaderNodeRGB")
    color2 = nodes.new(type="ShaderNodeRGB")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    out = nodes.new(type="ShaderNodeOutputMaterial")

    color1.outputs[0].default_value = (0.06, 0.06, 0.07, 1.0)
    color2.outputs[0].default_value = (0.18, 0.18, 0.19, 1.0)
    checker.inputs["Scale"].default_value = 1.6
    bsdf.inputs["Roughness"].default_value = 1.0

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    mat_bg.node_tree.nodes["Mapping"].inputs["Scale"].default_value = (1.0, 1.0, 1.0)
    bg_obj.data.materials.append(mat_bg)

    # Trapezoid Floor
    mesh = bpy.data.meshes.new("Trapezoid")
    obj = bpy.data.objects.new("Trapezoid", mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    verts = [
        (-bottom_width / 2, 0, -0.5),
        (bottom_width / 2, 0, -0.5),
        (top_width / 2, height, -0.5),
        (-top_width / 2, height, -0.5),
    ]
    bm_verts = [bm.verts.new(co) for co in verts]
    bm.faces.new(bm_verts)
    bm.to_mesh(mesh)
    bm.free()
    
    mat_surface = bpy.data.materials.new(name="SurfaceMat_Checker")
    mat_surface.use_nodes = True
    nodes = mat_surface.node_tree.nodes
    links = mat_surface.node_tree.links
    nodes.clear()
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    checker = nodes.new(type="ShaderNodeTexChecker")
    col_a = nodes.new(type="ShaderNodeRGB")
    col_b = nodes.new(type="ShaderNodeRGB")
    mixnode = nodes.new(type="ShaderNodeMixRGB")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    out = nodes.new(type="ShaderNodeOutputMaterial")
    
    checker.inputs["Scale"].default_value = 18.0
    col_a.outputs[0].default_value = (0.02, 0.02, 0.02, 1.0)
    col_b.outputs[0].default_value = (0.06, 0.06, 0.06, 1.0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], mixnode.inputs["Fac"])
    links.new(col_a.outputs["Color"], mixnode.inputs["Color1"])
    links.new(col_b.outputs["Color"], mixnode.inputs["Color2"])
    links.new(mixnode.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 1.0
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat_surface)

def calculate_object_mask(obj, scene, mask_save_path=None):
    original_engine = scene.render.engine
    original_film_transparent = scene.render.film_transparent
    original_use_compositing = scene.use_nodes
    original_view_transform = scene.view_settings.view_transform
    original_color_management = scene.view_settings.look
    original_materials = {}
    visibility_states = {}
    
    try:
        for o in bpy.data.objects:
            visibility_states[o] = o.hide_render
            if o == obj and o.data and hasattr(o.data, 'materials'):
                original_materials[o] = list(o.data.materials)
        
        for o in bpy.data.objects:
            o.hide_render = (o != obj)
        
        emission_mat = bpy.data.materials.new(name="TempEmissionMask")
        emission_mat.use_nodes = True
        emission_mat.node_tree.nodes.clear()
        nodes = emission_mat.node_tree.nodes
        links = emission_mat.node_tree.links
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs['Strength'].default_value = 1.0
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        if obj.data and hasattr(obj.data, 'materials'):
            obj.data.materials.clear()
            obj.data.materials.append(emission_mat)
        
        # We need EEVEE next or standard EEVEE for mask rendering speed
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except TypeError:
            scene.render.engine = 'BLENDER_EEVEE'
            
        scene.render.film_transparent = True
        scene.use_nodes = False
        scene.view_settings.view_transform = 'Raw'
        scene.view_settings.look = 'None'
        
        temp_path = os.path.join(job_temp_dir, f"temp_mask_{obj.name}_{random.randint(1000,9999)}.png")
        scene.render.filepath = temp_path
        bpy.ops.render.render(write_still=True)
        
        img = bpy.data.images.load(temp_path)
        pixels = np.array(img.pixels[:])
        width = img.size[0]
        height = img.size[1]
        pixels = pixels.reshape((height, width, 4))
        alpha_channel = pixels[:, :, 3]
        mask = alpha_channel > 0.5
        
        bpy.data.images.remove(img)
        os.remove(temp_path)
        
        if mask_save_path:
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            np.savez_compressed(mask_save_path, mask=mask)
            
    except Exception as e:
        print(f"Error calculating object mask: {e}")
        mask = np.zeros((scene.render.resolution_y, scene.render.resolution_x), dtype=bool)
    finally:
        for o, mats in original_materials.items():
            if o.data and hasattr(o.data, 'materials'):
                o.data.materials.clear()
                for mat in mats:
                    o.data.materials.append(mat)
        for o, state in visibility_states.items():
            o.hide_render = state
        scene.render.engine = original_engine
        scene.render.film_transparent = original_film_transparent
        scene.use_nodes = original_use_compositing
        scene.view_settings.view_transform = original_view_transform
        if hasattr(scene.view_settings, 'look'):
            scene.view_settings.look = original_color_management
        if 'emission_mat' in locals():
            bpy.data.materials.remove(emission_mat, do_unlink=True)
            
    return mask

def calculate_overlap_fraction(new_obj_mask, existing_obj_masks):
    if not existing_obj_masks:
        return 0.0
    new_pixel_count = np.sum(new_obj_mask)
    if new_pixel_count == 0:
        return 1.0 

    max_overlap_fraction = 0.0
    for existing_mask in existing_obj_masks:
        existing_pixel_count = np.sum(existing_mask)
        overlap = np.sum(new_obj_mask & existing_mask)
        denominator = min(new_pixel_count, existing_pixel_count)
        if denominator == 0:
            overlap_fraction = 1.0
        else:
            overlap_fraction = overlap / denominator
        if overlap_fraction > max_overlap_fraction:
            max_overlap_fraction = overlap_fraction
    return max_overlap_fraction

# ------------------ Placement logic ------------------
def try_generate_scene(scene, camera, object_specs, max_attempts_per_shape):
    placed_objects = []
    placed_bboxes = []
    used_positions_world = []
    placed_object_masks = []

    for obj_idx, spec in enumerate(object_specs):
        shape_type = spec["shape"]
        color_rgba = spec["color"]
        target_frac = spec["size"]
        placed = False
        
        for attempt in range(max_attempts_per_shape):
            y = random.uniform(edge_margin, height - edge_margin)
            width_at_y = bottom_width + ((top_width - bottom_width) * (y / height))
            half_width = width_at_y / 2
            x = random.uniform(-half_width + edge_margin, half_width - edge_margin)
            
            if not point_in_trapezoid(x, y):
                continue
            loc = (x, y, z_offset)

            obj_tmp = add_shape_primitive(shape_type, loc)
            if not obj_tmp:
                break
            bpy.context.view_layer.update()

            z_rotation = random.uniform(0, math.tau)
            obj_tmp.rotation_euler.z = z_rotation
            bpy.context.view_layer.update()
            
            target_pixels = target_frac * scene.render.resolution_y
            current_pixels = get_projected_height_pixels(obj_tmp, camera, scene)
            if current_pixels > 0:
                scale_factor = (target_pixels / current_pixels) ** 0.5
                obj_tmp.scale = (scale_factor, scale_factor, scale_factor)
                bpy.context.view_layer.update()

            bbox_world = get_world_bbox(obj_tmp)
            min_z = min(v.z for v in bbox_world)
            desired_bottom_z = z_offset
            if min_z < desired_bottom_z:
                lift = desired_bottom_z - min_z + 0.01
                obj_tmp.location.z += lift
                bpy.context.view_layer.update()

            bbox, coords2d = project_bbox_to_image(obj_tmp, camera, scene)
            xmin, ymin, xmax, ymax = bbox

            inter_xmin = max(0.0, xmin)
            inter_ymin = max(0.0, ymin)
            inter_xmax = min(1.0, xmax)
            inter_ymax = min(1.0, ymax)
            if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
                bpy.data.objects.remove(obj_tmp, do_unlink=True)
                continue

            if (xmin < image_overlap_margin or ymin < image_overlap_margin or
                xmax > 1.0 - image_overlap_margin or ymax > 1.0 - image_overlap_margin):
                bpy.data.objects.remove(obj_tmp, do_unlink=True)
                continue

            inside_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            total_area = normalized_bbox_area(bbox)
            if total_area == 0 or (inside_area / total_area) < 0.95:
                bpy.data.objects.remove(obj_tmp, do_unlink=True)
                continue

            new_radius = world_xy_footprint_radius(obj_tmp)
            world_too_close = False
            for (px, py, pradius) in used_positions_world:
                center_dist = math.hypot(px - x, py - y)
                if center_dist < (pradius + new_radius + min_distance_world):
                    world_too_close = True
                    break
                    
            if world_too_close:
                bpy.data.objects.remove(obj_tmp, do_unlink=True)
                continue
            
            obj_mask = calculate_object_mask(obj_tmp, scene)
            overlap_fraction = calculate_overlap_fraction(obj_mask, placed_object_masks)
            if overlap_fraction > allowed_overlap_fraction:
                bpy.data.objects.remove(obj_tmp, do_unlink=True)
                continue

            mat = create_material_for_color_rgba(color_rgba, name_hint=f"Mat_{shape_type}")
            if obj_tmp.data and isinstance(obj_tmp.data, bpy.types.Mesh):
                if len(obj_tmp.data.materials) == 0:
                    obj_tmp.data.materials.append(mat)
                else:
                    obj_tmp.data.materials[0] = mat

            placed_bboxes.append((xmin, ymin, xmax, ymax))
            used_positions_world.append((x, y, new_radius))
            placed_objects.append(obj_tmp)
            placed_object_masks.append(obj_mask)
            placed = True
            break

        if not placed:
            for o in placed_objects:
                try:
                    bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass
            return False, [], []

    placed_config = []
    for i, (obj_tmp, spec) in enumerate(zip(placed_objects, object_specs)):
        bbox, _ = project_bbox_to_image(obj_tmp, camera, scene)
        xmin, ymin, xmax, ymax = bbox
        final_mask = calculate_object_mask(obj_tmp, scene, mask_save_path=os.path.join(job_folder, f'obj{i}_mask.npz'))
        final_pixel_count = int(np.sum(final_mask))
        entry = {
            "shape_type": spec["shape"],
            "color_rgba": spec["color"],
            "requested_size_fraction": spec["size"],
            "uniform_scale": float(obj_tmp.scale.x),
            "world_location": [float(obj_tmp.location.x), float(obj_tmp.location.y), float(obj_tmp.location.z)],
            "image_bbox_norm": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "pixel_count": final_pixel_count
        }
        placed_config.append(entry)

    return True, placed_objects, placed_config

# ------------------ Lighting functions ------------------
def create_spot_light(name, location, rotation, energy_watts):
    light_data = bpy.data.lights.new(name=name, type='SPOT')
    light_data.energy = energy_watts
    light_data.use_shadow = True
    light_data.cycles.use_multiple_importance_sampling = True
    light_data.cycles.max_bounces = 1024
    light_data.spot_size = math.radians(45)

    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location
    light_obj.rotation_euler = tuple(radians(r) for r in rotation)

def create_point_light(name, location, energy_watts, radius_m):
    light_data = bpy.data.lights.new(name=name, type='POINT')
    light_data.energy = energy_watts
    light_data.shadow_soft_size = radius_m
    light_data.use_shadow = True
    light_data.cycles.use_multiple_importance_sampling = True
    light_data.cycles.max_bounces = 1024

    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = location

def main():
    try:
        clear_scene()
        setup_background_and_trapezoid()
        scene, camera = setup_scene_and_camera()

        # --- Create Lights ---
        create_point_light(name="Point", location=(0, 10, 5), energy_watts=30000, radius_m=15)
        create_spot_light(name="Spot_Right", location=(10, 25, 5), rotation=(-73.905, -16.028, 321.97), energy_watts=500)
        create_spot_light(name="Spot_Left", location=(-10, 25, 5), rotation=(-73.905, -16.0, 393.97), energy_watts=500)

        success = False
        for attempt_idx in range(attempts_per_image):
            ok, placed_objects, placed_config = try_generate_scene(
                scene, camera, object_specs, max_attempts_per_shape)
            if ok:
                success = True
                break

        if not success:
            meta = {
                "status": "failed",
                "reason": f"Could not place {len(object_specs)} objects after {attempts_per_image} attempts",
                "object_specs": object_specs,
                "repetition_index": repetition_index
            }
            with open(output_config, "w") as f:
                json.dump(meta, f)
            sys.exit(1)

        cfg_out = {
            "status": "ok",
            "object_specs": object_specs,
            "repetition_index": repetition_index,
            "seed": seed,
            "objects": placed_config,
            "camera_location": [float(v) for v in camera.location],
            "camera_rotation_euler": [float(v) for v in camera.rotation_euler],
            "image_resolution": image_resolution
        }
        with open(output_config, "w") as f:
            json.dump(cfg_out, f)

        # === RENDER SETTINGS ===
        bpy.context.scene.render.resolution_x = image_resolution[0]
        bpy.context.scene.render.resolution_y = image_resolution[1]
        bpy.context.scene.render.engine = 'CYCLES'
        prefs = bpy.context.preferences
        
        # Configure GPU Rendering
        try:
            prefs.addons['cycles'].preferences.compute_device_type = 'CUDA'
            prefs.addons['cycles'].preferences.get_devices()
            bpy.context.scene.cycles.device = 'GPU'
        except Exception:
            print("CUDA not available, falling back to CPU")
            bpy.context.scene.cycles.device = 'CPU'
            
        bpy.context.scene.cycles.samples = 512
        bpy.context.scene.cycles.use_shadows = False
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = output_image
        bpy.ops.render.render(write_still=True)
        sys.exit(0)

    except Exception as e:
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)