R"(
/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 */

const int g_TensorOperandsNonTemporal = 0x1;
const int g_TensorOperandsOutOfBoundsValue = 0x2;

#define _emu_GL_ARM_tensors_TypeSize_int8_t    1
#define _emu_GL_ARM_tensors_TypeSize_uint8_t   1
#define _emu_GL_ARM_tensors_TypeSize_bool      1 // bool is stored as int8_t
#define _emu_GL_ARM_tensors_TypeSize_int16_t   2
#define _emu_GL_ARM_tensors_TypeSize_uint16_t  2
#define _emu_GL_ARM_tensors_TypeSize_float16_t 2
#define _emu_GL_ARM_tensors_TypeSize_int       4
#define _emu_GL_ARM_tensors_TypeSize_uint      4
#define _emu_GL_ARM_tensors_TypeSize_float     4
#define _emu_GL_ARM_tensors_TypeSize_int64_t   8
#define _emu_GL_ARM_tensors_TypeSize_uint64_t  8
#define _emu_GL_ARM_tensors_TypeSize_double    8
#define _emu_GL_ARM_tensors_TypeSize(TYPE) (_emu_GL_ARM_tensors_TypeSize_ ## TYPE)

#define tensorSizeARM(tensor, dimension) uint(tensor.shape[dimension])

#define _emu_mvk_tensor_data(tensor) _tensorData##tensor.data

#define _emu_GL_ARM_tensors_read_array(tensor, coords, value, operands, outOfBoundsValue, TYPE) {                      \
  int64_t _emu_GL_ARM_tensors_offset = 0;                                                                              \
  bool _emu_GL_ARM_tensors_outOfBounds = false;                                                                        \
  for (int _emu_GL_ARM_tensors_i = 0; _emu_GL_ARM_tensors_i < coords.length(); ++_emu_GL_ARM_tensors_i) {              \
    if (int64_t(coords[_emu_GL_ARM_tensors_i]) >= tensor.shape[_emu_GL_ARM_tensors_i]) {                               \
      _emu_GL_ARM_tensors_outOfBounds = true;                                                                          \
      break;                                                                                                           \
    }                                                                                                                  \
    _emu_GL_ARM_tensors_offset += int64_t(coords[_emu_GL_ARM_tensors_i]) * tensor.stride[_emu_GL_ARM_tensors_i];       \
  }                                                                                                                    \
  _emu_GL_ARM_tensors_offset /= _emu_GL_ARM_tensors_TypeSize(TYPE);                                                    \
                                                                                                                       \
  for (int _emu_GL_ARM_tensors_i = 0; _emu_GL_ARM_tensors_i < value.length(); ++_emu_GL_ARM_tensors_i) {               \
    if (int64_t(coords[coords.length() - 1] + _emu_GL_ARM_tensors_i) >= tensor.shape[coords.length() - 1]) {           \
      _emu_GL_ARM_tensors_outOfBounds = true;                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    if (_emu_GL_ARM_tensors_outOfBounds) {                                                                             \
      value[_emu_GL_ARM_tensors_i] = TYPE(outOfBoundsValue);                                                           \
    } else {                                                                                                           \
      value[_emu_GL_ARM_tensors_i] =                                                                                   \
          TYPE(_emu_mvk_tensor_data(tensor)[uint(_emu_GL_ARM_tensors_offset + _emu_GL_ARM_tensors_i)]);                \
    }                                                                                                                  \
  }                                                                                                                    \
}

#define _emu_GL_ARM_tensors_read_scalar(tensor, coords, value, operands, outOfBoundsValue, TYPE) {                     \
  TYPE _emu_GL_ARM_tensors_valueArr[1];                                                                                \
  _emu_GL_ARM_tensors_read_array(tensor, coords, _emu_GL_ARM_tensors_valueArr, operands, outOfBoundsValue, TYPE);      \
  value = _emu_GL_ARM_tensors_valueArr[0];                                                                             \
}

#define _emu_GL_ARM_tensors_write_array(tensor, coords, value, operands, TYPE) {                                       \
  int64_t _emu_GL_ARM_tensors_offset = 0;                                                                              \
  bool _emu_GL_ARM_tensors_outOfBounds = false;                                                                        \
  for (int _emu_GL_ARM_tensors_i = 0; _emu_GL_ARM_tensors_i < coords.length(); ++_emu_GL_ARM_tensors_i) {              \
    if (int64_t(coords[_emu_GL_ARM_tensors_i]) >= tensor.shape[_emu_GL_ARM_tensors_i]) {                               \
      _emu_GL_ARM_tensors_outOfBounds = true;                                                                          \
      break;                                                                                                           \
    }                                                                                                                  \
    _emu_GL_ARM_tensors_offset += int64_t(coords[_emu_GL_ARM_tensors_i]) * tensor.stride[_emu_GL_ARM_tensors_i];       \
  }                                                                                                                    \
                                                                                                                       \
  _emu_GL_ARM_tensors_offset /= _emu_GL_ARM_tensors_TypeSize(TYPE);                                                    \
                                                                                                                       \
  for (int _emu_GL_ARM_tensors_i = 0; _emu_GL_ARM_tensors_i < value.length(); ++_emu_GL_ARM_tensors_i) {               \
    if (int64_t(coords[coords.length() - 1] + _emu_GL_ARM_tensors_i) >= tensor.shape[coords.length() - 1]) {           \
      _emu_GL_ARM_tensors_outOfBounds = true;                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    if (!_emu_GL_ARM_tensors_outOfBounds) {                                                                            \
      _emu_mvk_tensor_data(tensor)[uint(_emu_GL_ARM_tensors_offset + _emu_GL_ARM_tensors_i)] =                         \
          TYPE(value[_emu_GL_ARM_tensors_i]);                                                                          \
    }                                                                                                                  \
  }                                                                                                                    \
}

#define _emu_GL_ARM_tensors_write_scalar(tensor, coords, value, operands, TYPE) {                                      \
  TYPE _emu_GL_ARM_tensors_valueArr[] = {TYPE(value)};                                                                 \
  _emu_GL_ARM_tensors_write_array(tensor, coords, _emu_GL_ARM_tensors_valueArr, operands, TYPE);                       \
}

)"
