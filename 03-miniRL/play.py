
from ppo import Agent,Args,make_env
import tyro,torch
import gymnasium as gym

if __name__ == "__main__":
    args = tyro.cli(Args)
    es=[make_env(args.env_id, i, False, 'ppo-demo') for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(es)
    envs.envs[0].unwrapped.render_mode='human'
    agent = Agent(envs)
    agent.load_state_dict(torch.load('runs/ppo-demo.pt'))
    observations,_=envs.reset(seed=args.seed)
    for _ in range(100):
        #action = env.action_space.sample()  # this is where you would insert your policy
        with torch.no_grad():
            actions = agent.actor(torch.Tensor(observations)).cpu().numpy().argmax(-1)
        
        observations, rewards, terminateds, truncateds, infos = envs.step(actions)

        if terminateds.any() or truncateds.any():
            print('over')
            observations, infos = envs.reset()
        envs.envs[0].render()


    envs.close()
