// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdio>
#include <stdexcept>

namespace tensorrt_llm::common {
template <typename... Args>
[[noreturn]] inline void routingError(char const* format, Args... args) {
    char message[2048];
    std::snprintf(message, sizeof(message), format, args...);
    throw std::runtime_error(message);
}
}
#define TLLM_CHECK_WITH_INFO(condition, ...) do { if (!(condition)) tensorrt_llm::common::routingError(__VA_ARGS__); } while (0)
#define TLLM_LOG_ERROR(...) tensorrt_llm::common::routingError(__VA_ARGS__)
