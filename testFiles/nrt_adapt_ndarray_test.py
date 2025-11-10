#!/usr/bin/env python3
"""
Enhanced test script to trigger nrt_adapt_ndarray_to_python with debug output
"""

import numpy as np
import numba
from numba import njit
import sys

def test_basic_array_return():
    """Simple array return"""
    print("\n=== Test 1: Basic array return ===")
    @njit
    def make_array():
        return np.array([1, 2, 3, 4, 5])
    
    result = make_array()
    print(f"Result: {result}")

def test_multidimensional_arrays():
    """Different dimensional arrays"""
    print("\n=== Test 2: Multi-dimensional arrays ===")
    
    @njit
    def make_1d():
        return np.array([1, 2, 3])
    
    @njit  
    def make_2d():
        return np.array([[1, 2], [3, 4]])
    
    @njit
    def make_3d():
        return np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    print("1D array:")
    result1d = make_1d()
    print(f"  1D result: {result1d.shape}")
    
    print("2D array:")
    result2d = make_2d() 
    print(f"  2D result: {result2d.shape}")
    
    print("3D array:")
    result3d = make_3d()
    print(f"  3D result: {result3d.shape}")

def test_different_dtypes():
    """Arrays with different data types"""
    print("\n=== Test 3: Different data types ===")
    
    @njit
    def make_int_array():
        return np.array([1, 2, 3], dtype=np.int32)
    
    @njit
    def make_float_array():
        return np.array([1.1, 2.2, 3.3], dtype=np.float64)

    @njit
    def make_float32_array():
        return np.array([1.1, 2.2, 3.3], dtype=np.float32)
    
    @njit
    def make_bool_array():
        return np.array([True, False, True])

    @njit
    def make_complex64_array():
        return np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)

    @njit
    def make_complex128_array():
        return np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
    
    print("Int32 array:")
    int_arr = make_int_array()
    print(f"  Int array: {int_arr.dtype}\n")
    
    print("Float64 array:")
    float_arr = make_float_array()
    print(f"  Float array: {float_arr.dtype}\n")

    print("Float32 array:")
    float32_arr = make_float32_array()
    print(f"  Float32 array: {float32_arr.dtype}\n")
    
    print("Complex64 array:")
    complex64_arr = make_complex64_array()
    print(f"  Complex64 array: {complex64_arr.dtype}\n")

    print("Complex128 array:")
    complex128_arr = make_complex128_array()
    print(f"  Complex128 array: {complex128_arr.dtype}\n")

    print("Bool array:")
    bool_arr = make_bool_array()
    print(f"  Bool array: {bool_arr.dtype}\n")

def test_array_operations_return():
    """Arrays created through operations"""
    print("\n=== Test 4: Array operations return ===")
    
    @njit
    def array_operation():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        return a + b
    
    @njit
    def array_slice():
        arr = np.array([1, 2, 3, 4, 5])
        return arr[1:4]
    
    print("Array operation result:")
    op_result = array_operation()
    print(f"  Operation result: {op_result}")
    
    print("Array slice result:")
    slice_result = array_slice()
    print(f"  Slice result: {slice_result}")

def test_empty_arrays():
    """Edge case: empty arrays"""
    print("\n=== Test 5: Empty arrays ===")
    
    @njit
    def make_empty():
        return np.array([])
    
    @njit
    def make_zeros():
        return np.zeros(5)
    
    print("Empty array:")
    empty = make_empty()
    print(f"  Empty array: {empty.shape}")
    
    print("Zeros array:")
    zeros = make_zeros()
    print(f"  Zeros array: {zeros.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing nrt_adapt_ndarray_to_python with debug output")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Numba: {numba.__version__}")
    print(f"Platform: {sys.platform}")
    print("=" * 60)
    
    try:
        test_basic_array_return()
        test_multidimensional_arrays()
        test_different_dtypes()
        test_array_operations_return()
        test_empty_arrays()
        print("\n" + "=" * 60)
        print("All tests completed! Check debug output above.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        print("This might indicate the s390x integer size issue!")
        import traceback
        traceback.print_exc()
