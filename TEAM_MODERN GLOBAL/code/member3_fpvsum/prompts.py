"""Category mapping and CLIP text prompts (fixed a priori; never tuned on Samsung's evaluation videos)."""

# Samsung supported_classes.json label -> FPVSum category name -> descriptive phrases
CATEGORIES = [
    ("bike", "Bike", ["mountain biking", "riding a mountain bike on a trail"]),
    ("bikepolo", "Bikepolo", ["playing bike polo", "bicycle polo on a hard court"]),
    ("boxing", "Boxing", ["boxing", "sparring in a boxing ring"]),
    ("horse", "Horse", ["horseback riding", "riding a horse"]),
    ("jumping", "Jumping", ["skydiving", "wingsuit flying", "base jumping from a cliff"]),
    ("longboarding", "Longboarding", ["longboarding down a road", "riding a longboard skateboard downhill"]),
    ("motor", "Motor", ["riding a motorcycle", "motorbike riding on a road"]),
    ("parkour", "Parkour", ["parkour", "freerunning and jumping across rooftops"]),
    ("plane", "Plane", ["flying a small airplane", "piloting a plane, view from the cockpit"]),
    ("rockclimbing", "RockClimbing", ["rock climbing", "climbing a rock wall"]),
    ("scuba", "Scuba", ["scuba diving underwater", "diving near a coral reef"]),
    ("skate", "Skate", ["skateboarding", "skateboarding at a skatepark"]),
    ("ski", "Ski", ["skiing down a snowy slope", "skiing in the mountains"]),
    ("surfing", "Surfing", ["surfing a wave", "surfing in the ocean"]),
]
LABELS = [c[0] for c in CATEGORIES]

CATEGORY_TEMPLATES = [
    "a first-person video of {}.",
    "a GoPro point-of-view video of {}.",
    "a head-mounted camera view of {}.",
]

IMPORTANCE_POS_TEMPLATES = [
    "an exciting highlight moment of {}.",
    "the most interesting part of a first-person video of {}.",
    "an action-packed moment while {}.",
]
IMPORTANCE_NEG = [
    "a boring first-person video frame where nothing happens.",
    "a blurry, dark or static video frame.",
    "a camera pointing at the ground or the sky with no action.",
    "a person standing still waiting before the activity starts.",
]
