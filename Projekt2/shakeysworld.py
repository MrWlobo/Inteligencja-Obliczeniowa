from Projekt2.aipython.aipython.stripsProblem import STRIPS_domain, Planning_problem, Strips

def pickup_action(robot, ball, room, loc):
    return Strips(
        name=f"pickup_{ball}_at_{loc}",
        preconds={
            f"robot_at": loc,
            f"ball_at_{ball}": loc,
            "handempty": True,
            f"light_on_{room}": True
        },
        effects={
            f"holding_{robot}": ball,
            "handempty": False,
            f"ball_at_{ball}": "held"
        }
    )

def putdown_action(robot, ball, room, loc):
    return Strips(
        name=f"putdown_{ball}_at_{loc}",
        preconds={
            f"robot_at": loc,
            f"holding_{robot}": ball
        },
        effects={
            f"ball_at_{ball}": loc,
            f"holding_{robot}": None,
            "handempty": True
        }
    )

def move_action(robot, loc_from, loc_to, room_from, room_to, ball=None):
    preconds = {
        "robot_at": loc_from,
        f"light_on_{room_from}": True
    }

    effects = {
        "robot_at": loc_to
    }

    if ball:
        preconds[f"holding_{robot}"] = ball
        effects[f"ball_at_{ball}"] = loc_to

    return Strips(
        name=f"move_{robot}_{loc_from}_to_{loc_to}" + (f"_with_{ball}" if ball else ""),
        preconds=preconds,
        effects=effects
    )

def turnon_action(robot, room, switch_loc):
    return Strips(
        name=f"turnon_{room}_by_{robot}",
        preconds={
            "robot_at": switch_loc,
            "handempty": True,
            f"light_on_{room}": False
        },
        effects={
            f"light_on_{room}": True
        }
    )


features = {
    "robot_at": {"room1", "room2", "room3"},
    "ball_at_ball1" : {"room1", "room2", "room3", "held"},
    "ball_at_ball2" : {"room1", "room2", "room3", "held"},
    "handempty" : {True, False},
    "holding": {None, "ball1", "ball2"},
    "light_on_room1": {True, False},
    "light_on_room2": {True, False},
    "light_on_room3": {True, False},
}

robots = ["Shakey"]
rooms = ["room1", "room2", "room3"]
balls = ["ball1", "ball2"]
locations = ["loc1", "loc2", "loc3"]

actions = set()

for robot in robots:
    for ball in balls:
        for room, loc in zip(rooms, locations):
            actions.add(pickup_action(robot, ball, room, loc))
            actions.add(putdown_action(robot, ball, room, loc))

for robot in robots:
    for i, room_from in enumerate(rooms):
        loc_from = locations[i]
        for j, room_to in enumerate(rooms):
            if i == j:
                continue
            loc_to = locations[j]
            actions.add(move_action(robot, loc_from, loc_to, room_from, room_to))
            for ball in balls:
                actions.add(move_action(robot, loc_from, loc_to, room_from, room_to, ball))

for robot in robots:
    for i, room in enumerate(rooms):
        switch_loc = locations[i]
        actions.add(turnon_action(robot, room, switch_loc))

shakey_domain = STRIPS_domain(features, actions)

initial_state = {
    "robot_at": "room1",
    "handempty": True,
    "holding": None,
    "ball_at_ball1": "room1",
    "ball_at_ball2": "room2",
    "light_on_room1": True,
    "light_on_room2": False,
    "light_on_room3": False
}

goal_state = {
    "light_on_room3": True,
    "ball_at_ball1": "room3",
    "ball_at_ball2": "room3"
}

problem = Planning_problem(shakey_domain, initial_state, goal_state)