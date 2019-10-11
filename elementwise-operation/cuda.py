import ctypes
import numpy


class Vector:
    """Elementwise Vector Operations
    """
    def __init__(self):
        self.elementWise = ctypes.cdll.LoadLibrary('./libelementwise.so')

    def Add(self, A, B):
        """Elementwise Vector Addition
        Args:
            A (numpy.ndarray): Vector A
            B (numpy.ndarray): Vector B
        Returns:
            C (numpy.ndarray): Elementwise addition
        """

        if not(len(A) == len(B)):
            raise ValueError('Length of both vectors should be same')

        self.elementWise.VectorAdd.argtypes = \
            [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_int]
        self.elementWise.VectorAdd.restype =\
            ctypes.POINTER(ctypes.c_double)

        A = (ctypes.c_double * len(A))(*A)
        B = (ctypes.c_double * len(B))(*B)
        C = self.elementWise.VectorAdd(A, B, len(A))
        return numpy.ctypeslib.as_array(C, shape=(len(A), ))

    def Sub(self, A, B):
        """Elementwise Vector Subtraction
        Args:
            A (numpy.ndarray): Vector A
            B (numpy.ndarray): Vector B
        Returns:
            C (numpy.ndarray): Elementwise subtraction
        """

        if not(len(A) == len(B)):
            raise ValueError('Length of both vectors should be same')

        self.elementWise.VectorSub.argtypes = \
            [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_int]
        self.elementWise.VectorSub.restype =\
            ctypes.POINTER(ctypes.c_double)

        A = (ctypes.c_double * len(A))(* A)
        B = (ctypes.c_double * len(B))(* B)
        C = self.elementWise.VectorSub(A, B, len(A))
        return numpy.ctypeslib.as_array(C, shape=(len(A), ))

    def Mul(self, A, B):
        """Elementwise Vector Multiplication
        Args:
            A (numpy.ndarray): Vector A
            B (numpy.ndarray): Vector B
        Returns:
            C (numpy.ndarray): Elementwise multiplication
        """

        if not(len(A) == len(B)):
            raise ValueError('Length of both vectors should be same')

        self.elementWise.VectorMul.argtypes = \
            [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.c_int]
        self.elementWise.VectorMul.restype =\
            ctypes.POINTER(ctypes.c_double)

        A = (ctypes.c_double * len(A))(* A)
        B = (ctypes.c_double * len(B))(* B)
        C = self.elementWise.VectorMul(A, B, len(A))
        return numpy.ctypeslib.as_array(C, shape=(len(A), ))
