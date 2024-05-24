import time
import torch
import numpy as np

BAG_NUM = 10


def trans_fileName_to_npz():
    start_time = time.time()
    list_str_path = [
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-162021_0.bag.2ba4e5d23f5007cc82f234b8f0fc1061.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-192636_0.bag.a844dc03230a81d0e402ba50866b99a6.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-195031_0.bag.b318970397ad08ed5d9151ad9986a298.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230509-201217_0.bag.0bbd27a1c38ca54329aec9e3e079552b.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-105631_0.bag.032aa42e53472a3b820a58076af2216d.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-110036_0.bag.087eef6ce5f40d887acadab8b667890a.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-114328_0.bag.dbe5e6d057a8eacc97bc41f5065f200c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-120023_0.bag.dc6adfe75aa8ca27c55619c11d54e41c.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-143837_0.bag.adfda910b229b40ff9abdb8428d85a57.npz",
        "./data/raw/PL004_event_ddp_expert_event_20230510-144653_0.bag.eb352e856a29fc7b20f17737779e11b1.npz",
    ]
    list_str_path = list_str_path[:BAG_NUM]
    list_npz_data = []
    for str_path in list_str_path:
        npz_data = np.load(str_path, allow_pickle=False)
        list_npz_data.append(npz_data)
    print("file name 2 npz time per bag (ms): ", (time.time() - start_time) * 1000 / BAG_NUM)
    return list_npz_data


def trans_npz_to_tensor(list_npz_data):
    start_time = time.time()
    tensor_bag_vectornet_object_feature = torch.zeros(BAG_NUM, 254, 100, 16, 11).cuda()
    tensor_bag_vectornet_object_mask = torch.zeros(BAG_NUM, 254, 100, 16).cuda()
    tensor_bag_vectornet_static_feature = torch.zeros(BAG_NUM, 254, 80, 16, 6).cuda()
    tensor_bag_ego_gt_traj = torch.zeros(BAG_NUM, 254, 20, 2).cuda()
    tensor_bag_ego_gt_traj_hist = torch.zeros(BAG_NUM, 254, 10, 2).cuda()
    tensor_bag_ego_gt_traj_long = torch.zeros(BAG_NUM, 254, 60, 2).cuda()
    for npz_i in range(len(list_npz_data)):
        npz_data = list_npz_data[npz_i]  # 1ms

        numpy_vectornet_object_feature = npz_data['vectornet_object_feature']  # 32ms
        tensor_vectornet_object_feature = torch.from_numpy(numpy_vectornet_object_feature).cuda()  # 1ms
        npz_timestep = tensor_vectornet_object_feature.size(0)
        tensor_bag_vectornet_object_feature[npz_i, :npz_timestep] = tensor_vectornet_object_feature  # 1ms
        tensor_bag_vectornet_object_feature[npz_i, :npz_timestep] = tensor_vectornet_object_feature  # 1ms

        numpy_vectornet_object_mask = npz_data['vectornet_object_mask']  # 3ms
        tensor_vectornet_object_mask = torch.from_numpy(numpy_vectornet_object_mask).cuda()  # 3ms
        tensor_bag_vectornet_object_mask[npz_i, :npz_timestep] = tensor_vectornet_object_mask  # 1ms

        numpy_vectornet_static_feature = npz_data['vectornet_static_feature']  # 14ms
        tensor_vectornet_static_feature = torch.from_numpy(numpy_vectornet_static_feature).cuda()  # 5ms
        tensor_bag_vectornet_static_feature[npz_i, :npz_timestep] = tensor_vectornet_static_feature  # 1ms

        numpy_ego_gt_traj = npz_data['ego_gt_traj']
        tensor_ego_gt_traj = torch.from_numpy(numpy_ego_gt_traj).cuda()  # 3ms
        tensor_bag_ego_gt_traj[npz_i, :npz_timestep] = tensor_ego_gt_traj  # 1ms

        numpy_ego_gt_traj_hist = npz_data['ego_gt_traj_hist']
        tensor_ego_gt_traj_hist = torch.from_numpy(numpy_ego_gt_traj_hist).cuda()  # 2.5ms
        tensor_bag_ego_gt_traj_hist[npz_i, :npz_timestep] = tensor_ego_gt_traj_hist  # 1ms

        numpy_ego_gt_traj_long = npz_data['ego_gt_traj_long']
        tensor_ego_gt_traj_long = torch.from_numpy(numpy_ego_gt_traj_long).cuda()  # 3ms
        tensor_bag_ego_gt_traj_long[npz_i, :npz_timestep] = tensor_ego_gt_traj_long  # 1ms
    print("npz 2 tensor time per bag (ms): ", (time.time() - start_time) * 1000 / BAG_NUM)
    return (tensor_bag_ego_gt_traj, tensor_bag_ego_gt_traj_hist, tensor_bag_ego_gt_traj_long,
            tensor_bag_vectornet_object_feature, tensor_bag_vectornet_object_mask, tensor_bag_vectornet_static_feature)


def main():
    # file name 2 npz
    list_npz_data = trans_fileName_to_npz()

    # npz 2 numpy
    (tensor_bag_ego_gt_traj, tensor_bag_ego_gt_traj_hist, tensor_bag_ego_gt_traj_long,
     tensor_bag_vectornet_object_feature, tensor_bag_vectornet_object_mask, tensor_bag_vectornet_static_feature) = (
        trans_npz_to_tensor(list_npz_data))

    print(tensor_bag_vectornet_object_feature.size())
    print(tensor_bag_vectornet_object_mask.size())
    print(tensor_bag_vectornet_static_feature.size())
    print(tensor_bag_ego_gt_traj.size())
    print(tensor_bag_ego_gt_traj_hist.size())
    print(tensor_bag_ego_gt_traj_long.size())

    print("Finished...")


if __name__ == '__main__':
    main()
