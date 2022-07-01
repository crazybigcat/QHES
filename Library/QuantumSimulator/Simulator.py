import random
from collections import Counter

import numpy as np
import torch as tc

from Library.QuantumSimulator.Circuit import Circuit


# [1, 0] is |0>, [0, 1] is |1>
# do not use fake functions, it will be removed soon
# please apply gate on position as list(range(?)) as much as possible, this will make it faster
# please control gate on position as list(range(?, n_qubit)) as much as possible, this will make it faster

class SimulatorProcess:
    def __init__(self, n_qubit, device='cuda:0', dtype=tc.complex64, rand_seed=1):
        self.n_qubit = n_qubit
        self.device = device
        self.dtype = dtype
        self.rand_seed = rand_seed
        self._state = None
        self.initialize_state()
        self.tmp_gate = None
        self.shape = (2,) * self.n_qubit
        self.chi = 2 ** self.n_qubit
        self.hamiltonian = None
        self.inverse = False
        self.circuit = Circuit(n_qubit, device=device, dtype=dtype)

    def initialize_state(self):
        self._state = tc.zeros(2 ** self.n_qubit, device=self.device, dtype=self.dtype)
        self._state[0] = 1
        self._state = self._state.view((2,) * self.n_qubit)

    def simulate(self, clear_circuit=True):
        if len(self.circuit) > 0:
            for cc in self.circuit:
                self.act_single_gate(cc)
        if clear_circuit:
            self.circuit = Circuit(self.n_qubit, self.device, self.dtype)

    @property
    def state(self):
        return self._state.clone()

    @state.setter
    def state(self, state):
        self._state = state.clone().reshape(self.shape).to(self.device).to(self.dtype)

    def act_single_gate(self, gate):
        gate.regularize(self.n_qubit)
        # gate can be sparse, but there seems to be no speedup
        # one should be careful when add inverse gates
        m_p = len(gate.position)
        m_c = len(gate.control)
        old_position = gate.position + gate.control
        new_position = list(range(m_p)) + list(range(-m_c, 0))
        if gate.inverse:
            tmp_gate = gate.tensor.conj().t()
        else:
            tmp_gate = gate.tensor
        self._state = self._state.movedim(old_position, new_position).contiguous().view(2 ** m_p, -1, 2 ** m_c)

        # self._state[:, :, -1] = tmp_gate.mm(self._state[:, :, -1])
        # The reason to introduce tmp_state is for the auto grad
        tmp_state = self._state.clone().detach()
        tmp_state[:, :, -1] = tmp_gate.mm(self._state[:, :, -1])
        tmp_state[:, :, :-1] = self._state[:, :, :-1]
        self.state = tmp_state

        self._state = self._state.view(self.shape).movedim(new_position, old_position)

    def sampling(self, n_shots=1024, position=None, basis=None, if_print=True, rand_seed=None):
        if rand_seed is None:
            rand_seed = self.rand_seed
        if rand_seed is not None:
            random.seed(rand_seed)
        if basis is not None:
            tmp_state = self.change_measure_basis(position, basis)
        else:
            tmp_state = self._state.clone().detach()
        if position is None:
            position = list(range(self.n_qubit))
            weight = tc.abs(tmp_state.contiguous().view(-1)) ** 2
            m_p = len(position)
        else:
            m_p = len(position)
            state_tmp = tmp_state.movedim(position, list(range(m_p))).contiguous().view(2 ** m_p, -1)
            weight = tc.abs(tc.einsum('ab,ba->a', state_tmp, state_tmp.conj().t()))

        population = list()
        for pp in range(2 ** m_p):
            element = bin(pp)[2:]
            element = (m_p - len(element)) * '0' + element
            population.append(element)

        res = Counter(random.choices(population, weight, k=n_shots))
        if if_print:
            for key in res.keys():
                print(key, res[key])
        return res

    @staticmethod
    def count_sample(res, ss, position, if_print=True):
        new_res = dict()
        for key in res.keys():
            flag = True
            for pp in position:
                if key[pp] != ss[position.index(pp)]:
                    flag = False
            if flag:
                new_res[key] = res[key]
        if if_print:
            for key in new_res.keys():
                print(key, new_res[key])
        return new_res

    def change_measure_basis(self, position, basis):
        if position is None:
            position = list(range(self.n_qubit))
        x_basis = tc.tensor([[1, 1], [1, -1]], device=self.device, dtype=self.dtype) / np.sqrt(2)
        y_basis = tc.tensor([[1, 1j], [1, -1j]], device=self.device, dtype=self.dtype) / np.sqrt(2)
        tmp_state = self._state.clone().detach()
        for nn in range(len(position)):
            pp = position[nn]
            if basis[nn] == 'x':
                tmp_state = tc.einsum('abc,bd->adc', tmp_state.reshape(2 ** pp, -1, 2 ** (self.n_qubit - pp - 1)), x_basis)
            elif basis[nn] == 'y':
                tmp_state = tc.einsum('abc,bd->adc', tmp_state.reshape(2 ** pp, -1, 2 ** (self.n_qubit - pp - 1)), y_basis)
            tmp_state = tmp_state.view(self.shape).contiguous()
        return tmp_state

    def collapse(self, position, basis=None):
        m_p = len(position)
        if basis is None:
            basis = '0' * m_p
        if m_p != len(basis):
            raise ValueError('error in extend, check position')
        index = int(basis, 2)
        new_position = list(range(-m_p, 0))
        tmp_state = self.state.movedim(position, new_position).reshape(-1, 2 ** m_p)
        return tmp_state[:, index]

    def fake_local_measure(self, position, operator):

        assert position == sorted(position), 'warning!!! The position should be in descending order.'
        index_contract = list(range(self.n_qubit))
        for pp in position:
            index_contract.remove(pp)
        reduce_rho = tc.tensordot(self._state, self._state.conj(), dims=[index_contract, index_contract])
        measure_result = tc.einsum('ab,ba->', reduce_rho, operator)
        # print(reduce_rho)
        return measure_result

    def fake_local_rho(self, position):

        assert position == sorted(position), 'warning!!! The position should be in descending order.'
        index_contract = list(range(self.n_qubit))
        for pp in position:
            index_contract.remove(pp)
        reduce_rho = tc.tensordot(self._state, self._state.conj(), dims=[index_contract, index_contract])
        return reduce_rho.reshape(2**len(position), -1)

