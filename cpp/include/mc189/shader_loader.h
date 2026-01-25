#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <optional>
#include <memory>
#include <functional>

namespace minecraft_vulkan {

// Shader names matching the 26 compute shaders
enum class ShaderType {
    AabbOps,
    BlockBreaking,
    BlockPlacing,
    BlockUpdates,
    Crafting,
    CrystalTick,
    DimensionTeleport,
    DragonAi,
    DragonCombat,
    DragonDeath,
    EndTerrain,
    Experience,
    EyeOfEnder,
    FindStrongholds,
    GameTick,
    HealthRegen,
    HungerTick,
    InventoryOps,
    MobAiBase,
    MobAiBlaze,
    MobAiEnderman,
    MobSpawning,
    NetherTerrain,
    OverworldTerrain,
    PortalTick,
    StatusEffects,
    COUNT
};

// Convert shader type to filename (without extension)
const char* shader_type_to_name(ShaderType type);

// Forward declaration for include handler
class ShaderIncludeHandler;

// Compilation options
struct ShaderCompileOptions {
    bool optimize = true;                      // Enable optimization passes
    bool generate_debug_info = false;          // Include debug symbols
    std::vector<std::string> defines;          // Preprocessor defines
    std::vector<std::filesystem::path> include_dirs;  // Include search paths
};

// Result of shader compilation
struct CompilationResult {
    bool success = false;
    std::vector<uint32_t> spirv;
    std::string error_message;
    std::string warnings;
};

// SPIR-V source type
enum class SpirvSource {
    PreCompiled,   // Loaded from .spv file
    RuntimeCompiled // Compiled at runtime via shaderc
};

// Shader module with metadata
struct ShaderModuleInfo {
    VkShaderModule module = VK_NULL_HANDLE;
    ShaderType type;
    SpirvSource source;
    std::string source_path;
    size_t spirv_size = 0;  // Size in bytes
};

// Custom include handler for shaderc
class ShaderIncludeHandler {
public:
    explicit ShaderIncludeHandler(const std::vector<std::filesystem::path>& include_dirs);

    // Resolve include path and return content
    std::optional<std::string> resolve(const std::string& requested_source,
                                       const std::string& requesting_source,
                                       bool is_relative);

    // Get resolved path (for error messages)
    std::optional<std::filesystem::path> get_resolved_path(const std::string& name) const;

private:
    std::vector<std::filesystem::path> include_dirs_;
    std::unordered_map<std::string, std::filesystem::path> resolved_cache_;
};

// Main shader loader class
class ShaderLoader {
public:
    // Initialize with Vulkan device and shader directory
    ShaderLoader(VkDevice device, const std::filesystem::path& shader_dir);
    ~ShaderLoader();

    // Non-copyable, movable
    ShaderLoader(const ShaderLoader&) = delete;
    ShaderLoader& operator=(const ShaderLoader&) = delete;
    ShaderLoader(ShaderLoader&& other) noexcept;
    ShaderLoader& operator=(ShaderLoader&& other) noexcept;

    // Load all 26 shaders (precompiled .spv preferred, fallback to runtime compilation)
    bool load_all_shaders(const ShaderCompileOptions& options = {});

    // Load a specific shader
    bool load_shader(ShaderType type, const ShaderCompileOptions& options = {});

    // Get shader module by type
    VkShaderModule get_module(ShaderType type) const;

    // Get shader module info
    const ShaderModuleInfo* get_module_info(ShaderType type) const;

    // Get module by string name
    VkShaderModule get_module_by_name(const std::string& name) const;

    // Check if shader is loaded
    bool is_loaded(ShaderType type) const;

    // Check if all shaders are loaded
    bool all_loaded() const;

    // Reload a shader (e.g., for hot-reloading during development)
    bool reload_shader(ShaderType type, const ShaderCompileOptions& options = {});

    // Unload a shader
    void unload_shader(ShaderType type);

    // Unload all shaders
    void unload_all();

    // Compile GLSL to SPIR-V (can be used standalone)
    static CompilationResult compile_glsl_to_spirv(
        const std::string& source,
        const std::string& filename,
        const ShaderCompileOptions& options = {});

    // Load precompiled SPIR-V from file
    static std::optional<std::vector<uint32_t>> load_spirv_file(const std::filesystem::path& path);

    // Get list of loaded shaders
    std::vector<ShaderType> get_loaded_shaders() const;

    // Get shader directory
    const std::filesystem::path& get_shader_directory() const { return shader_dir_; }

    // Batch compile all shaders to .spv files (for offline compilation)
    bool compile_all_to_files(const std::filesystem::path& output_dir,
                              const ShaderCompileOptions& options = {});

private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::filesystem::path shader_dir_;
    std::unordered_map<ShaderType, ShaderModuleInfo> modules_;
    std::unordered_map<std::string, ShaderType> name_to_type_;

    // Create VkShaderModule from SPIR-V bytecode
    VkShaderModule create_shader_module(const std::vector<uint32_t>& spirv);

    // Try to load precompiled SPIR-V, fallback to runtime compilation
    std::optional<std::vector<uint32_t>> load_or_compile(
        ShaderType type,
        const ShaderCompileOptions& options,
        SpirvSource& out_source);

    // Read shader source file
    static std::optional<std::string> read_source_file(const std::filesystem::path& path);

    // Initialize name mapping
    void init_name_mapping();
};

// Utility: Get all shader types
std::vector<ShaderType> get_all_shader_types();

// Utility: Create pipeline shader stage info
VkPipelineShaderStageCreateInfo make_shader_stage_info(
    VkShaderModule module,
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT,
    const char* entry_point = "main");

} // namespace minecraft_vulkan
