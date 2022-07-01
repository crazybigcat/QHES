import torch as tc
import numpy as np

import copy


class Gate:
    def __init__(self, tensor, position=None, control=None, inverse=False, independent=True):
        if isinstance(tensor, Gate):
            if independent:
                self.tensor = tensor.tensor.clone().detach()
            else:
                self.tensor = tensor.tensor
            if tensor.position is None:
                self.position = None
            else:
                self.position = tensor.position.copy()
            if tensor.control is None:
                self.control = None
            else:
                self.control = tensor.control.copy()
            self.inverse = tensor.inverse
            if not ((position is None) and (control is None) and (not inverse)):
                print('waring in initializing gate')
        else:
            if independent:
                self.tensor = tensor.clone().detach()
            else:
                self.tensor = tensor
            self.position = position
            if control is None:
                control = []
            self.control = control
            self.inverse = inverse

    @property
    def requires_grad(self):
        return self.tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad):
        self.tensor.requires_grad = requires_grad

    @property
    def grad(self):
        return self.tensor.grad

    @grad.setter
    def grad(self, value):
        self.tensor.grad = value

    def inv(self):
        self.inverse = not self.inverse

    def square(self):
        if self.tensor.is_sparse:
            self.tensor = tc.sparse.mm(self.tensor, self.tensor)
        else:
            self.tensor = self.tensor.mm(self.tensor)

    def controlled_gate(self, n_control=1, output=False):
        # this function has not been tested
        c_gate = self.tensor.clone().detach()
        a_m = tc.tensor([[1, 0], [0, 0]], device=self.tensor.device, dtype=self.tensor.dtype)
        b_m = tc.tensor([[0, 0], [0, 1]], device=self.tensor.device, dtype=self.tensor.dtype)
        new_shape = c_gate.shape
        for nn in range(n_control):
            n_dim = round(c_gate.numel() ** 0.5)
            eye_m = tc.eye(n_dim, dtype=self.tensor.dtype, device=self.tensor.device)
            new_shape = new_shape + (2, 2)
            c_gate = tc.kron(a_m, eye_m) + tc.kron(b_m, c_gate.view(n_dim, n_dim))
        if output:
            return c_gate.view(new_shape)
        else:
            self.tensor = c_gate

    def to(self, device_or_dtype):
        self.tensor = self.tensor.to(device_or_dtype)
        return self

    def regularize(self, n_qubit):
        self.position = list(map(lambda x: x % n_qubit, self.position))
        self.control = list(map(lambda x: x % n_qubit, self.control))


def rand_gate(dim=2, position=None, control=None, device='cpu', dtype=tc.complex64, requires_grad=False):
    # Haar random gate
    tmp_tensor = tc.randn(dim, dim, device=device, dtype=dtype)
    q, r = tc.linalg.qr(tmp_tensor)
    sign_matrix = tc.sign(tc.real(tc.diag(r)))
    gate = tc.einsum('ij,j->ij', q, sign_matrix)
    gate = Gate(gate, position, control, False)
    gate.requires_grad = requires_grad
    return gate
    

def hadamard(position=None, control=None, device='cpu', dtype=tc.complex64):
    gate = tc.tensor([[1, 1], [1, -1]], device=device, dtype=dtype) / np.sqrt(2)
    return Gate(gate, position, control, False)


def phase_shift(theta, position=None, control=None, device='cpu', dtype=tc.complex64):
    gate = tc.eye(2, device=device, dtype=dtype)
    gate[1, 1] = tc.exp(tc.tensor(theta * 1j))
    return Gate(gate, position, control, False)


def not_gate(position=None, control=None, device='cpu', dtype=tc.complex64):
    gate = tc.tensor([[0, 1], [1, 0]], device=device, dtype=dtype)
    return Gate(gate, position, control, False)


def swap_gate(position=None, control=None, device='cpu', dtype=tc.complex64):
    gate = tc.zeros(4, 4, device=device, dtype=dtype)
    gate[0, 0] = 1
    gate[1, 2] = 1
    gate[2, 1] = 1
    gate[3, 3] = 1
    return Gate(gate, position, control, False)


def time_evolution(hamiltonian, time, position=None, control=None, device='cpu', dtype=tc.complex64):
    hamiltonian = hamiltonian.to(device).to(dtype)
    if hamiltonian.is_sparse:
        gate = tc.matrix_exp(-1j * hamiltonian.to_dense() * time)
        gate = gate.to_sparse()
    else:
        gate = tc.matrix_exp(-1j * hamiltonian * time)
    return Gate(gate, position, control, False)