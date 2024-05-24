import torch

from utils_agent import Agent


def main():
    agent = Agent()

    state_dict = torch.load("./data/interim/state_dict_temp.pt", map_location=torch.device("cuda:0"))
    agent.load_state_dict(state_dict)

    print("Finished")


if __name__ == '__main__':
    main()
