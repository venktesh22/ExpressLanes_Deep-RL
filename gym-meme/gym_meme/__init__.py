from gym.envs.registration import register

register(
    id='meme-v0',
    entry_point='gym_meme.envs:MemeEnv',
)
