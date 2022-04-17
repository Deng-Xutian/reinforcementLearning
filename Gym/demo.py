import gym

class demo():
    def __init__(self):
        pass

    def main(self):
        env = gym.make('CarRacing-v1')
        env.reset()
        print(env.action_space)
        for epoch in range(1000):
            env.render()
            env.step(env.action_space.sample())

    def all_envs(self):
        print(gym.envs.registry)

if __name__ == "__main__":
    a = demo()
    a.main()
    # a.all_envs()
