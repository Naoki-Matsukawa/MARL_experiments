from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageDraw


ALLY_COLOR = (55, 125, 245)
ENEMY_COLOR = (220, 70, 70)
DEAD_COLOR = (120, 120, 120)
TEXT_COLOR = (30, 30, 30)
SIGHT_COLOR = (80, 150, 255)
BACKGROUND = (246, 247, 249)
GRID_COLOR = (225, 228, 232)


def _unit_xy(unit):
    return float(unit.pos.x), float(unit.pos.y)


def _world_to_pixel(x, y, map_x, map_y, width, height, margin):
    scale_x = (width - 2 * margin) / max(float(map_x), 1.0)
    scale_y = (height - 2 * margin) / max(float(map_y), 1.0)
    px = margin + x * scale_x
    py = height - margin - y * scale_y
    return px, py, min(scale_x, scale_y)


def _draw_grid(draw, width, height, margin, step=64):
    for x in range(margin, width - margin + 1, step):
        draw.line([(x, margin), (x, height - margin)], fill=GRID_COLOR)
    for y in range(margin, height - margin + 1, step):
        draw.line([(margin, y), (width - margin, y)], fill=GRID_COLOR)


def _draw_health_bar(draw, px, py, unit, radius):
    health_max = max(float(getattr(unit, "health_max", 1.0)), 1.0)
    health = max(float(getattr(unit, "health", 0.0)), 0.0)
    ratio = min(health / health_max, 1.0)
    bar_w = radius * 2.4
    bar_h = 4
    left = px - bar_w / 2
    top = py - radius - 9
    draw.rectangle([left, top, left + bar_w, top + bar_h], fill=(180, 180, 180))
    draw.rectangle([left, top, left + bar_w * ratio, top + bar_h], fill=(65, 175, 85))


def _draw_unit(draw, unit, unit_id, color, map_x, map_y, width, height, margin):
    x, y = _unit_xy(unit)
    px, py, scale = _world_to_pixel(x, y, map_x, map_y, width, height, margin)
    alive = getattr(unit, "health", 0) > 0
    fill = color if alive else DEAD_COLOR
    radius = max(5, min(9, int(scale * 0.65)))
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=(20, 20, 20))
    draw.text((px + radius + 2, py - radius), str(unit_id), fill=TEXT_COLOR)
    if alive:
        _draw_health_bar(draw, px, py, unit, radius)


def _draw_sight(draw, unit, sight_range, map_x, map_y, width, height, margin):
    if getattr(unit, "health", 0) <= 0:
        return
    x, y = _unit_xy(unit)
    px, py, scale = _world_to_pixel(x, y, map_x, map_y, width, height, margin)
    r = sight_range * scale
    draw.ellipse(
        [px - r, py - r, px + r, py + r],
        outline=SIGHT_COLOR,
        width=1,
    )


def render_smac_topdown(env, width=512, height=512, show_sight=False):
    margin = 28
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, width, height, margin)
    draw.rectangle([margin, margin, width - margin, height - margin], outline=(160, 165, 170))

    map_x = getattr(env, "map_x", 0) or 1
    map_y = getattr(env, "map_y", 0) or 1

    if show_sight:
        for agent_id, unit in env.agents.items():
            _draw_sight(draw, unit, env.unit_sight_range(agent_id), map_x, map_y, width, height, margin)

    for enemy_id, unit in env.enemies.items():
        _draw_unit(draw, unit, enemy_id, ENEMY_COLOR, map_x, map_y, width, height, margin)

    for agent_id, unit in env.agents.items():
        _draw_unit(draw, unit, agent_id, ALLY_COLOR, map_x, map_y, width, height, margin)

    draw.text((margin, 8), "blue: allies  red: enemies", fill=TEXT_COLOR)
    draw.text((width - 120, 8), "step {}".format(getattr(env, "_episode_steps", 0)), fill=TEXT_COLOR)
    return np.asarray(image, dtype=np.uint8)
