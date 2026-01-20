objects = [
    "basin",
    "bathroom_product",
    "bed",
    "bedding",
    "blanket",
    "book",
    "bowl",
    "cabinet",
    "ceiling",
    "ceiling_light",
    "chair",
    "chandelier",
    "chopstick",
    "clock",
    "closestool",
    "computer",
    "cookware",
    "cosmetic",
    "cup",
    "curtain",
    "cushion",
    "daily_equipment",
    "decorative_box",
    "desk",
    "dining_table",
    "door",
    "door_handle",
    "doorsill",
    "electric_appliance",
    "floor",
    "floor_lamp",
    "flower",
    "fork",
    "fridge",
    "fruit",
    "hardware_decoration",
    "kettle",
    "kitchenware",
    "knife",
    "menorah",
    "microwave",
    "mirror",
    "office_supply",
    "ornament",
    "other_cooker",
    "painting",
    "picture_frame",
    "pillar",
    "pillow",
    "plate",
    "range_hood",
    "screen",
    "shelf",
    "sofa",
    "spoon",
    "spot_light",
    "stool",
    "storage",
    "table",
    "tablecloth",
    "table_lamp",
    "tea_set",
    "television",
    "throw_pillow",
    "tooling",
    "toy",
    "tray",
    "unknown",
    "vase",
    "wall",
    "wall_decoration",
    "wall_light",
    "washing_machine",
    "water_tap",
    "window",
    "wine_set",
]

import csv
import numpy as np
import matplotlib.colors as mcolors

# orignal random colormap
# with open("interior_agent_objects.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["object", "r", "g", "b"])
#     for obj in objects:
#         color = np.random.rand(3)  # random RGB in [0,1]
#         writer.writerow([obj, *color])


# css4 colormap
css4_colors = list(mcolors.CSS4_COLORS.values())
np.random.seed(43)  # optional, for reproducibility
css4_colors = np.random.permutation(css4_colors)

with open("colormap_css4.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["object", "r", "g", "b"])
    for i, obj in enumerate(objects):
        rgb = mcolors.to_rgb(css4_colors[i])
        writer.writerow([obj, *rgb])

# plot colormap
import matplotlib.pyplot as plt

colormap = {}
with open("colormap_css4.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        object = row["object"]
        r = float(row["r"])
        g = float(row["g"])
        b = float(row["b"])
        colormap[object] = [r, g, b]
fig, ax = plt.subplots(figsize=(6, 12))
for i, (obj, color) in enumerate(colormap.items()):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
    ax.text(1.1, i + 0.5, obj, va="center", fontsize=8)
ax.set_xlim(0, 3)
ax.set_ylim(0, len(colormap))
ax.axis("off")

plt.show()