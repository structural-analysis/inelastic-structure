all = [
        # "3story-static-inelastic-softening",    # still not fixed after sifting
        # "plate-semiconfined-inelastic",  # possible-bifurcation # wierd behavior on multiple runs # infinite loop in sifting
        # "3d-simple-beam-static-inelastic",  # possible bifurcation

        # STATIC EXAMPLES:
        # "truss-kassimali-115-static-elastic",
        "simple-beam-static-elastic",
        "simple-beam-static-inelastic",
        "torre-static-elastic",
        "skew-beams-static-elastic",
        "tripod-corner",
        "tripod-unload",
        "3story-static-elastic",
        "3story-static-inelastic-perfect",
        "3d-2side-static-elastic",
        "3d-simple-beam-static-elastic",
        "3d-2side-static-inelastic",
        "wall-1element-elastic",
        "wall-1element-elastic-q8r",
        "wall-9element-elastic",
        "wall-4element-elastic-q8r",
        "wall-1element-inelastic",
        "wall-4element-inelastic-q8r",
        "plate-1element-elastic-q8r",
        "plate-4element-elastic-q8r",
        "plate-9element-confined-elastic",
        "plate-confined-elastic",
        "plate-semiconfined-elastic",
        "plate-1element-inelastic-q8r",
        "plate-check-inelastic-q8r",
        "plate-square-inelastic-q8r",
        "plate-4element-inelastic-q8r",
        "plate-9element-confined-inelastic",
        "plate-confined-inelastic",

        # DYNAMIC EXAMPLES:
        "3d-simple-beam-dynamic-inelastic",
        "simple-beam-dynamic-elastic",
        "torre-dynamic-elastic",
        "1story-dynamic-elastic",
        "2story-dynamic-elastic",
        "3d-2side-dynamic-elastic",
        "3d-simple-beam-dynamic-elastic",
        "simple-beam-dynamic-inelastic-1phase",
        "2story-dynamic-inelastic",
        "1story-dynamic-inelastic-ll1.0-ap400k",  # infinite loop in sifting when minimum selected pieces used
        "3d-2side-dynamic-inelastic",  # infinite loop in sifting when minimum selected pieces used
    ]

statics = [analysis for analysis in all if "-static" in analysis]
statics = statics + [
    "wall-1element-elastic",
    "wall-1element-elastic-q8r",
    "wall-9element-elastic",
    "wall-4element-elastic-q8r",
    "wall-1element-inelastic",
    "wall-4element-inelastic-q8r",
    "plate-1element-elastic-q8r",
    "plate-4element-elastic-q8r",
    "plate-9element-confined-elastic",
    "plate-confined-elastic",
    "plate-semiconfined-elastic",
    "plate-1element-inelastic-q8r",
    "plate-check-inelastic-q8r",
    "plate-square-inelastic-q8r",
    "plate-4element-inelastic-q8r",
    "plate-9element-confined-inelastic",
    "plate-confined-inelastic",
    "tripod-corner",
    "tripod-unload",
]
dynamics = [analysis for analysis in all if "-dynamic" in analysis]
two_d_frames = [
    "simple-beam-static-elastic",
    "simple-beam-static-inelastic",
    "torre-static-elastic",
    "skew-beams-static-elastic",
    "tripod-corner",
    "tripod-unload",
    "3story-static-elastic",
    "3story-static-inelastic-perfect",
    "simple-beam-dynamic-elastic",
    "torre-dynamic-elastic",
    "1story-dynamic-elastic",
    "2story-dynamic-elastic",
    "simple-beam-dynamic-inelastic-1phase",
    "2story-dynamic-inelastic",
    "1story-dynamic-inelastic-ll1.0-ap400k",
]
three_d_frames = [analysis for analysis in all if "-3d" in analysis]
plates = [analysis for analysis in all if "-plate" in analysis]
walls = [analysis for analysis in all if "-wall" in analysis]
trusses = [analysis for analysis in all if "-truss" in analysis]
softenings = [analysis for analysis in all if "-softening" in analysis]
elastics = [analysis for analysis in all if "-elastic" in analysis]
inelastics = [analysis for analysis in all if "-inelastic" in analysis]
inelastics = inelastics + [
    "tripod-corner",
    "tripod-unload", 
]


class Examples:
    all: list = all
    statics: list = statics
    dynamics: list = dynamics
    two_d_frames: list = two_d_frames
    three_d_frames: list = three_d_frames
    plates: list = plates
    walls: list = walls
    trusses: list = trusses
    softenings: list = softenings
    elastics: list = elastics
    inelastics: list = inelastics
