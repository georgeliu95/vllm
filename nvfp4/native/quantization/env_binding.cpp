// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone environment binding for the SM103-only experiment.
#include "tensorrt_llm/common/config.h"
#include <cstdlib>
#include <stdexcept>
#include <string_view>

TRTLLM_NAMESPACE_BEGIN
namespace common
{
bool getEnvEnablePDL()
{
    static bool const enabled = [] {
        auto const* value = std::getenv("TRTLLM_ENABLE_PDL");
        if (value == nullptr || std::string_view(value) == "1")
            return true;
        if (std::string_view(value) == "0")
            return false;
        throw std::invalid_argument("TRTLLM_ENABLE_PDL must be 0 or 1");
    }();
    return enabled;
}
} // namespace common
TRTLLM_NAMESPACE_END
