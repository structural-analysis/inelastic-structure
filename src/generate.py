import os
import numpy as np
from src.settings import settings
from src.models import Material, Section, FrameElement2D

examples_dir = "input/examples/"
example_name = settings.example_name

global_cords_path = os.path.join(examples_dir, example_name, "global_cords.csv")
materials_path = os.path.join(examples_dir, example_name, "materials.csv")
sections_path = os.path.join(examples_dir, example_name, "elements/sections.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")


def get_global_cords():
    cords = []
    cords_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(cords_array.shape[0]):
        cords.append(cords_array[i])
    return cords


def generate_materials():
    materials = {}
    materials_array = np.loadtxt(fname=materials_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(materials_array.shape[0]):
        materials[materials_array[i, 0]] = Material(name=materials_array[i, 0])
    return materials


def generate_sections(materials):
    sections = {}
    sections_array = np.loadtxt(fname=sections_path, usecols=range(6), delimiter=",", ndmin=2, skiprows=1, dtype=str)
    for i in range(sections_array.shape[0]):
        sections[sections_array[i, 0]] = Section(
            material=materials[sections_array[i, 1]],
            a=float(sections_array[i, 2]),
            ix=float(sections_array[i, 3]),
            iy=float(sections_array[i, 4]),
            zp=float(sections_array[i, 5]),
        )
    return sections


def generate_frames():
    global_cords = get_global_cords()
    materials = generate_materials()
    sections = generate_sections(materials)
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)

    frames = []
    for i in range(frames_array.shape[0]):
        frames.append(
            FrameElement2D(
                section=sections[frames_array[i, 0]],
                start=global_cords[int(frames_array[i, 1])],
                end=global_cords[int(frames_array[i, 2])],
                ends_fixity=frames_array[i, 3]
            )
        )
    return frames
