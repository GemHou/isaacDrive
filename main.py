import numpy as np


def main():
    path = "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz"
    data = np.load(path, allow_pickle=False)

    print("Finished...")


if __name__ == '__main__':
    main()
