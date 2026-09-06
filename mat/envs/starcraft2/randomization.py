from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import debug_pb2 as d_pb


def _in_bounds(env, x, y):
    if x < 0 or y < 0 or x >= env.map_x or y >= env.map_y:
        return False
    return env.check_bounds(int(x), int(y))


def _sample_offset(radius):
    if radius <= 0:
        return 0.0, 0.0
    angle = random.uniform(0.0, 2.0 * math.pi)
    distance = random.uniform(0.0, radius)
    return math.cos(angle) * distance, math.sin(angle) * distance


def _candidate_group_positions(env, units, radius, attempts):
    if radius <= 0:
        return [(unit.pos.x, unit.pos.y) for unit in units]

    for _ in range(max(attempts, 1)):
        dx, dy = _sample_offset(radius)
        positions = [(unit.pos.x + dx, unit.pos.y + dy) for unit in units]
        if all(_in_bounds(env, x, y) for x, y in positions):
            return positions

    return [(unit.pos.x, unit.pos.y) for unit in units]


def _candidate_unit_positions(env, units, radius, attempts):
    positions = []
    for unit in units:
        x, y = unit.pos.x, unit.pos.y
        for _ in range(max(attempts, 1)):
            dx, dy = _sample_offset(radius)
            candidate_x = unit.pos.x + dx
            candidate_y = unit.pos.y + dy
            if _in_bounds(env, candidate_x, candidate_y):
                x, y = candidate_x, candidate_y
                break
        positions.append((x, y))
    return positions


def randomize_enemy_positions(env):
    if not getattr(env, "randomize_enemy_position", False):
        return

    radius = float(getattr(env, "enemy_position_jitter", 0.0))
    if radius <= 0:
        return

    enemies = [env.enemies[i] for i in sorted(env.enemies.keys())]
    if not enemies:
        return

    attempts = int(getattr(env, "enemy_position_jitter_attempts", 20))
    mode = getattr(env, "enemy_position_jitter_mode", "group")
    if mode == "unit":
        positions = _candidate_unit_positions(env, enemies, radius, attempts)
    else:
        positions = _candidate_group_positions(env, enemies, radius, attempts)

    kill_tags = [unit.tag for unit in enemies if unit.health > 0]
    commands = []
    if kill_tags:
        commands.append(d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=kill_tags)))

    for unit, (x, y) in zip(enemies, positions):
        commands.append(
            d_pb.DebugCommand(
                create_unit=d_pb.DebugCreateUnit(
                    unit_type=unit.unit_type,
                    owner=2,
                    pos=sc_common.Point2D(x=x, y=y),
                    quantity=1,
                )
            )
        )

    env._controller.debug(commands)
    env._controller.step(1)
    env._obs = env._controller.observe()
    refreshed = [
        unit
        for unit in env._obs.observation.raw_data.units
        if unit.owner == 2
    ]
    env.enemies = {i: unit for i, unit in enumerate(refreshed)}
