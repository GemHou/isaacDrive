import torch

from utils_agent import Agent

DEVICE = torch.device("cpu")  # cuda:0 cpu


def main():
    agent = Agent()

    state_dict = torch.load("./data/interim/state_dict_temp.pt", map_location=DEVICE)
    agent.load_state_dict(state_dict)

    print("Finished")


if __name__ == '__main__':
    main()
