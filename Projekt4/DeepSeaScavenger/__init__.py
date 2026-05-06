from gymnasium.envs.registration import register

register(
    id="DeepSeaScavenger/GridWorld-v0",
    entry_point="DeepSeaScavenger.envs:GridWorldEnv",
)
