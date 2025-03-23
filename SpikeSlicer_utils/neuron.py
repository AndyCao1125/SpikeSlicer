from spikingjelly.activation_based import neuron
import copy


class IFNode(neuron.IFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class LIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])