import numpy as np
import torch as tc

import copy

from Library.QuantumSimulator import Gate
from Library.BasicFunSZZ import tensor_trick as tt


# [1, 0] is |0>, [0, 1] is |1>
# do not use fake functions, it will be removed soon
# please apply gate on position as list(range(?)) as much as possible, this will make it faster
# please control gate on position as list(range(?, n_qubit)) as much as possible, this will make it faster

class Circuit(list):

    def __init__(self, n_qubit, device='cpu', dtype=tc.complex64):
        list.__init__(self)
        self.n_qubit = n_qubit
        self.device = device
        self.dtype = dtype

    @property
    def requires_grad(self):
        flag = list()
        for gg in self:
            flag.append(gg.requires_grad)
        return flag

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        if isinstance(requires_grad, bool):
            for gg in self:
                gg.requires_grad = requires_grad
        else:
            for nn in range(len(self)):
                self[nn].requires_grad = requires_grad[nn]

    def compose(self, circuit, position=None, control=None, inverse=False):
        if isinstance(circuit, Circuit):
            tmp_circuit = copy.deepcopy(circuit)
            if inverse:
                tmp_circuit.inv()
            self.extend(tmp_circuit, position, control)
        elif isinstance(circuit, Gate.Gate):
            self.add_single_gate(circuit, position, control, inverse)
        else:
            raise ValueError('input is not Gate or Circuit')

    def extend(self, circuit, position=None, control=None):
        if control is None:
            control = []
        if position is None:
            if self.n_qubit == circuit.n_qubit:
                for cc in circuit:
                    self.append(cc)
            else:
                raise ValueError('error in extend, position is needed')
        elif len(position) == circuit.n_qubit:
            for cc in circuit:
                new_p = list()
                new_c = list()
                for oo in cc.position:
                    new_p.append(position[oo])
                for oo in cc.control:
                    new_c.append(position[oo])
                for oo in control:
                    new_c.append(oo)
                cc.position = new_p.copy()
                cc.control = new_c.copy()
                self.append(cc)
        else:
            raise ValueError('error in extend, check position')

    def __add__(self, circuit):
        if circuit.n_qubit == self.n_qubit:
            return super().__add__(circuit)
        else:
            raise ValueError('error in +, n_qubit not equal')

    def add_single_gate(self, gate, position=None, control=None, inverse=False):
        # gate can be sparse, but there seems to be no speedup
        # one should be careful when add inverse gates
        tmp_gate = Gate.Gate(gate)
        if tt.have_same_iterable(position, control):
            raise ValueError('position and control have same qubits')
        if position is not None:
            tmp_gate.position = position
        if tmp_gate.position is None:
            raise ValueError('position should not be None')
        if control is not None:
            tmp_gate.control = tmp_gate.control + control
        if inverse:
            tmp_gate.inv()
        self.append(tmp_gate)

    def inv(self):
        self.reverse()
        for gate in self:
            gate.inv()

    def to(self, device_or_dtype):
        for cc in self:
            cc.to(device_or_dtype)
        return self

    def square(self):
        for gate in self:
            gate.square()

    def hadamard(self, position, control=None):
        for pp in position:
            tmp_gate = Gate.hadamard([pp], control, self.device, self.dtype)
            self.add_single_gate(tmp_gate)

    def phase_shift(self, theta=tc.pi, position=None, control=None):
        for pp in position:
            tmp_gate = Gate.phase_shift(theta, [pp], control, self.device, self.dtype)
            self.add_single_gate(tmp_gate)

    def not_gate(self, position=None, control=None):
        for pp in position:
            tmp_gate = Gate.not_gate([pp], control, self.device, self.dtype)
            self.add_single_gate(tmp_gate)

    def rand_gate(self, dim, position=None, control=None, requires_grad=False):
        tmp_gate = Gate.rand_gate(dim, position, control, self.device, self.dtype, requires_grad=requires_grad)
        self.add_single_gate(tmp_gate)

    def swap_gate(self, position=None, control=None):
        if len(position) != 2:
            print('wrong use')
        else:
            tmp_gate = Gate.swap_gate(position, control, self.device, self.dtype)
            self.add_single_gate(tmp_gate)

    def time_evolution(self, hamiltonian, time, position=None, control=None):
        tmp_gate = Gate.time_evolution(hamiltonian, time, position, control, self.device, self.dtype)
        self.add_single_gate(tmp_gate)

    def qft(self, position, control=None, inverse=False):
        if control is None:
            control = []
        tmp_circuit = qft(self.n_qubit, position, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def ch(self, unitary, position_phi, position_c, control=None, inverse=False):
        tmp_circuit = ch(self.n_qubit, unitary, position_phi, position_c, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def qpe(self, unitary, position_phi, position_qpe, control=None, inverse=False):
        tmp_circuit = qpe(self.n_qubit, unitary, position_phi, position_qpe, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def add_one(self, position=None, control=None, inverse=False):
        tmp_circuit = add_one(self.n_qubit, position, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def qhc(self, unitary, position_phi, position_qpe, position_f, n_f=None,
            control=None, inverse=False):
        tmp_circuit = qhc(self.n_qubit, unitary, position_phi, position_qpe, position_f,
                          n_f, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def quantum_coin(self, unitary, position_phi, position_coin, control=None, inverse=False):
        tmp_circuit = quantum_coin(self.n_qubit, unitary, position_phi, position_coin,
                                   control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)

    def qdc(self, unitary, position_phi, position_coin, position_f,
            n_f=None, control=None, inverse=False):
        tmp_circuit = qdc(self.n_qubit, unitary, position_phi, position_coin, position_f,
                          n_f, control, inverse, self.device, self.dtype)
        self.extend(tmp_circuit)


def qft(n_qubit, position, control=None, inverse=False, device='cpu', dtype=tc.complex64):
    if control is None:
        control = []
    tmp_circuit = Circuit(n_qubit, device, dtype)
    m_qft = len(position)
    perm = list(range(m_qft))
    perm.reverse()
    theta_list = []
    theta = 2 * np.pi
    if inverse:
        theta = -theta
    for mm in range(m_qft + 1):
        theta_list.append(theta)
        theta = theta / 2
    for mm in range(m_qft):
        tmp_circuit.hadamard([position[mm]])
        for nn in range(mm + 1, m_qft):
            tmp_circuit.phase_shift(theta_list[nn - mm + 1], [position[mm]], control=[position[nn]] + control)
    for mm in range(m_qft // 2):
        tmp_circuit.swap_gate([position[mm], position[- mm - 1]])
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def ch(n_qubit, unitary, position_phi, position_c, control=None, inverse=False, device='cpu', dtype=tc.complex64):
    if control is None:
        control = []
    tmp_circuit = Circuit(n_qubit, device, dtype)
    if isinstance(unitary, Gate.Gate):
        tmp_gate = Gate.Gate(unitary)
    elif isinstance(unitary, Circuit):
        tmp_gate = copy.deepcopy(unitary)
    m_fch = len(position_c)
    for mm in range(m_fch):
        tmp_circuit.compose(tmp_gate, position_phi, [position_c[- mm - 1]] + control, inverse)
        tmp_gate.square()
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def qpe(n_qubit, unitary, position_phi, position_qpe, control=None, inverse=False, device='cpu', dtype=tc.complex64):
    if control is None:
        control = []
    tmp_circuit = Circuit(n_qubit, device, dtype)
    tmp_circuit.hadamard(position_qpe)
    tmp_circuit.ch(unitary, position_phi, position_qpe, control, False)
    tmp_circuit.qft(position_qpe, control=control, inverse=True)
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def qhc(n_qubit, unitary, position_phi, position_qpe, position_f, n_f=None,
        control=None, inverse=False, device='cpu', dtype=tc.complex64):
    tmp_circuit = Circuit(n_qubit, device, dtype)
    if control is None:
        control = []
    if n_f is None:
        n_f = 2 ** (len(position_f) - 1) - 1
    if n_f > (2 ** len(position_f) - 1) / 2:
        print('warning, n_f is too large')
    for nn in range(n_f):
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.qpe(unitary, position_phi, position_qpe,
                        control=position_f + control, inverse=False)
        # sample(1000, position_qpe)
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.add_one(position_f, [position_qpe[0]] + control, inverse=False)
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.qpe(unitary, position_phi, position_qpe,
                        control=position_f + control, inverse=True)
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.add_one(position_f, control=control, inverse=False)
        tmp_circuit.not_gate(position_qpe, control)
        tmp_circuit.add_one(position_f, control=position_qpe + control, inverse=True)
        tmp_circuit.not_gate(position_qpe, control)
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def add_one(n_qubit, position=None, control=None, inverse=False, device='cpu', dtype=tc.complex64):
    tmp_circuit = Circuit(n_qubit, device, dtype)
    if control is None:
        control = []
    if position is None:
        position = list(range(n_qubit))
    m_a = len(position)
    for mm in range(m_a):
        tmp_circuit.not_gate([position[mm]], control=position[mm + 1:] + control)
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def quantum_coin(n_qubit, unitary, position_phi, position_coin, control=None, inverse=False,
                 device='cpu', dtype=tc.complex64):
    tmp_circuit = Circuit(n_qubit, device, dtype)
    if control is None:
        control = []
    tmp_circuit.hadamard(position_coin, control)
    tmp_circuit.compose(unitary, position_phi, position_coin)
    tmp_circuit.not_gate(position_coin, control)
    tmp_circuit.compose(unitary, position_phi, position_coin, True)
    # self.phase_shift(-eig_value, position_coin, control)
    tmp_circuit.not_gate(position_coin, control)
    tmp_circuit.hadamard(position_coin, control)
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit


def qdc(n_qubit, unitary, position_phi, position_coin, position_f, n_f=None,
        control=None, inverse=False, device='cpu', dtype=tc.complex64):
    tmp_circuit = Circuit(n_qubit, device, dtype)
    if control is None:
        control = []
    if n_f is None:
        n_f = 2 ** (len(position_f) - 1) - 1
    if n_f > (2 ** len(position_f) - 1):
        print('warning, n_f is too large')
    for nn in range(n_f):
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.quantum_coin(unitary, position_phi, position_coin, control=position_f + control, inverse=False)
        tmp_circuit.not_gate(position_f, control)
        tmp_circuit.add_one(position_f, position_coin + control, inverse=False)
    if inverse:
        tmp_circuit.inv()
    return tmp_circuit
