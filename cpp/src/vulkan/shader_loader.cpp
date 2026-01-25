#include "shader_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

// Check if shaderc is available
#if __has_include(<shaderc/shaderc.hpp>)
#define HAS_SHADERC 1
#include <shaderc/shaderc.hpp>
#else
#define HAS_SHADERC 0
#endif

namespace minecraft_vulkan {

// Shader name lookup table
static const char* SHADER_NAMES[] = {
    "aabb_ops",
    "block_breaking",
    "block_placing",
    "block_updates",
    "crafting",
    "crystal_tick",
    "dimension_teleport",
    "dragon_ai",
    "dragon_combat",
    "dragon_death",
    "end_terrain",
    "experience",
    "eye_of_ender",
    "find_strongholds",
    "game_tick",
    "health_regen",
    "hunger_tick",
    "inventory_ops",
    "mob_ai_base",
    "mob_ai_blaze",
    "mob_ai_enderman",
    "mob_spawning",
    "nether_terrain",
    "overworld_terrain",
    "portal_tick",
    "status_effects"
};

static_assert(sizeof(SHADER_NAMES) / sizeof(SHADER_NAMES[0]) == static_cast<size_t>(ShaderType::COUNT),
              "SHADER_NAMES count must match ShaderType::COUNT");

const char* shader_type_to_name(ShaderType type) {
    size_t idx = static_cast<size_t>(type);
    if (idx >= static_cast<size_t>(ShaderType::COUNT)) {
        return nullptr;
    }
    return SHADER_NAMES[idx];
}

std::vector<ShaderType> get_all_shader_types() {
    std::vector<ShaderType> types;
    types.reserve(static_cast<size_t>(ShaderType::COUNT));
    for (size_t i = 0; i < static_cast<size_t>(ShaderType::COUNT); ++i) {
        types.push_back(static_cast<ShaderType>(i));
    }
    return types;
}

VkPipelineShaderStageCreateInfo make_shader_stage_info(
    VkShaderModule module,
    VkShaderStageFlagBits stage,
    const char* entry_point)
{
    VkPipelineShaderStageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage = stage;
    info.module = module;
    info.pName = entry_point;
    return info;
}

// ShaderIncludeHandler implementation
ShaderIncludeHandler::ShaderIncludeHandler(const std::vector<std::filesystem::path>& include_dirs)
    : include_dirs_(include_dirs) {}

std::optional<std::string> ShaderIncludeHandler::resolve(
    const std::string& requested_source,
    const std::string& requesting_source,
    bool is_relative)
{
    std::filesystem::path resolved;

    if (is_relative && !requesting_source.empty()) {
        // Relative include: search relative to requesting file first
        std::filesystem::path requesting_dir = std::filesystem::path(requesting_source).parent_path();
        std::filesystem::path candidate = requesting_dir / requested_source;
        if (std::filesystem::exists(candidate)) {
            resolved = std::filesystem::canonical(candidate);
        }
    }

    // If not found via relative path, search include directories
    if (resolved.empty()) {
        for (const auto& dir : include_dirs_) {
            std::filesystem::path candidate = dir / requested_source;
            if (std::filesystem::exists(candidate)) {
                resolved = std::filesystem::canonical(candidate);
                break;
            }
        }
    }

    if (resolved.empty()) {
        return std::nullopt;
    }

    // Cache the resolution
    resolved_cache_[requested_source] = resolved;

    // Read file content
    std::ifstream file(resolved);
    if (!file) {
        return std::nullopt;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::optional<std::filesystem::path> ShaderIncludeHandler::get_resolved_path(const std::string& name) const {
    auto it = resolved_cache_.find(name);
    if (it != resolved_cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ShaderLoader implementation
ShaderLoader::ShaderLoader(VkDevice device, const std::filesystem::path& shader_dir)
    : device_(device), shader_dir_(shader_dir)
{
    init_name_mapping();
}

ShaderLoader::~ShaderLoader() {
    unload_all();
}

ShaderLoader::ShaderLoader(ShaderLoader&& other) noexcept
    : device_(other.device_)
    , shader_dir_(std::move(other.shader_dir_))
    , modules_(std::move(other.modules_))
    , name_to_type_(std::move(other.name_to_type_))
{
    other.device_ = VK_NULL_HANDLE;
}

ShaderLoader& ShaderLoader::operator=(ShaderLoader&& other) noexcept {
    if (this != &other) {
        unload_all();
        device_ = other.device_;
        shader_dir_ = std::move(other.shader_dir_);
        modules_ = std::move(other.modules_);
        name_to_type_ = std::move(other.name_to_type_);
        other.device_ = VK_NULL_HANDLE;
    }
    return *this;
}

void ShaderLoader::init_name_mapping() {
    for (size_t i = 0; i < static_cast<size_t>(ShaderType::COUNT); ++i) {
        ShaderType type = static_cast<ShaderType>(i);
        name_to_type_[SHADER_NAMES[i]] = type;
    }
}

bool ShaderLoader::load_all_shaders(const ShaderCompileOptions& options) {
    bool all_success = true;
    for (const auto& type : get_all_shader_types()) {
        if (!load_shader(type, options)) {
            std::cerr << "Failed to load shader: " << shader_type_to_name(type) << "\n";
            all_success = false;
        }
    }
    return all_success;
}

bool ShaderLoader::load_shader(ShaderType type, const ShaderCompileOptions& options) {
    if (is_loaded(type)) {
        return true;
    }

    SpirvSource source_type;
    auto spirv_opt = load_or_compile(type, options, source_type);
    if (!spirv_opt) {
        return false;
    }

    VkShaderModule module = create_shader_module(*spirv_opt);
    if (module == VK_NULL_HANDLE) {
        return false;
    }

    ShaderModuleInfo info;
    info.module = module;
    info.type = type;
    info.source = source_type;
    info.source_path = (shader_dir_ / (std::string(shader_type_to_name(type)) + ".comp")).string();
    info.spirv_size = spirv_opt->size() * sizeof(uint32_t);

    modules_[type] = info;
    return true;
}

VkShaderModule ShaderLoader::get_module(ShaderType type) const {
    auto it = modules_.find(type);
    if (it == modules_.end()) {
        return VK_NULL_HANDLE;
    }
    return it->second.module;
}

const ShaderModuleInfo* ShaderLoader::get_module_info(ShaderType type) const {
    auto it = modules_.find(type);
    if (it == modules_.end()) {
        return nullptr;
    }
    return &it->second;
}

VkShaderModule ShaderLoader::get_module_by_name(const std::string& name) const {
    auto it = name_to_type_.find(name);
    if (it == name_to_type_.end()) {
        return VK_NULL_HANDLE;
    }
    return get_module(it->second);
}

bool ShaderLoader::is_loaded(ShaderType type) const {
    return modules_.find(type) != modules_.end();
}

bool ShaderLoader::all_loaded() const {
    return modules_.size() == static_cast<size_t>(ShaderType::COUNT);
}

bool ShaderLoader::reload_shader(ShaderType type, const ShaderCompileOptions& options) {
    unload_shader(type);
    return load_shader(type, options);
}

void ShaderLoader::unload_shader(ShaderType type) {
    auto it = modules_.find(type);
    if (it != modules_.end()) {
        if (device_ != VK_NULL_HANDLE && it->second.module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, it->second.module, nullptr);
        }
        modules_.erase(it);
    }
}

void ShaderLoader::unload_all() {
    for (auto& [type, info] : modules_) {
        if (device_ != VK_NULL_HANDLE && info.module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, info.module, nullptr);
        }
    }
    modules_.clear();
}

std::vector<ShaderType> ShaderLoader::get_loaded_shaders() const {
    std::vector<ShaderType> loaded;
    loaded.reserve(modules_.size());
    for (const auto& [type, info] : modules_) {
        loaded.push_back(type);
    }
    return loaded;
}

VkShaderModule ShaderLoader::create_shader_module(const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv.size() * sizeof(uint32_t);
    create_info.pCode = spirv.data();

    VkShaderModule module;
    if (vkCreateShaderModule(device_, &create_info, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

std::optional<std::vector<uint32_t>> ShaderLoader::load_or_compile(
    ShaderType type,
    const ShaderCompileOptions& options,
    SpirvSource& out_source)
{
    const char* name = shader_type_to_name(type);
    if (!name) {
        return std::nullopt;
    }

    // Try precompiled SPIR-V first
    std::filesystem::path spv_path = shader_dir_ / (std::string(name) + ".spv");
    if (std::filesystem::exists(spv_path)) {
        auto spirv = load_spirv_file(spv_path);
        if (spirv) {
            out_source = SpirvSource::PreCompiled;
            return spirv;
        }
    }

    // Fallback to runtime compilation
    std::filesystem::path comp_path = shader_dir_ / (std::string(name) + ".comp");
    auto source = read_source_file(comp_path);
    if (!source) {
        std::cerr << "Failed to read shader source: " << comp_path << "\n";
        return std::nullopt;
    }

    // Add shader directory to include paths
    ShaderCompileOptions compile_opts = options;
    compile_opts.include_dirs.insert(compile_opts.include_dirs.begin(), shader_dir_);

    auto result = compile_glsl_to_spirv(*source, comp_path.string(), compile_opts);
    if (!result.success) {
        std::cerr << "Failed to compile shader " << name << ": " << result.error_message << "\n";
        return std::nullopt;
    }

    if (!result.warnings.empty()) {
        std::cerr << "Shader " << name << " warnings: " << result.warnings << "\n";
    }

    out_source = SpirvSource::RuntimeCompiled;
    return std::move(result.spirv);
}

std::optional<std::string> ShaderLoader::read_source_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        return std::nullopt;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::optional<std::vector<uint32_t>> ShaderLoader::load_spirv_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }

    auto size = file.tellg();
    if (size <= 0 || size % sizeof(uint32_t) != 0) {
        return std::nullopt;
    }

    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);

    if (!file) {
        return std::nullopt;
    }

    // Basic SPIR-V validation: check magic number
    if (spirv.empty() || spirv[0] != 0x07230203) {
        return std::nullopt;
    }

    return spirv;
}

#if HAS_SHADERC

// Custom shaderc includer that handles damage.glsl and other includes
class ShadercIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    explicit ShadercIncluder(std::shared_ptr<ShaderIncludeHandler> handler)
        : handler_(std::move(handler)) {}

    shaderc_include_result* GetInclude(
        const char* requested_source,
        shaderc_include_type type,
        const char* requesting_source,
        size_t /*include_depth*/) override
    {
        bool is_relative = (type == shaderc_include_type_relative);
        auto content = handler_->resolve(requested_source, requesting_source, is_relative);

        auto result = new shaderc_include_result;

        if (content) {
            // Store content in a way that persists
            auto& stored = stored_content_.emplace_back(std::move(*content));
            auto& stored_name = stored_names_.emplace_back(requested_source);

            result->source_name = stored_name.c_str();
            result->source_name_length = stored_name.size();
            result->content = stored.c_str();
            result->content_length = stored.size();
            result->user_data = nullptr;
        } else {
            // Return error
            auto& error_msg = stored_content_.emplace_back("Include not found: " + std::string(requested_source));
            result->source_name = "";
            result->source_name_length = 0;
            result->content = error_msg.c_str();
            result->content_length = error_msg.size();
            result->user_data = nullptr;
        }

        return result;
    }

    void ReleaseInclude(shaderc_include_result* data) override {
        delete data;
    }

private:
    std::shared_ptr<ShaderIncludeHandler> handler_;
    std::vector<std::string> stored_content_;
    std::vector<std::string> stored_names_;
};

CompilationResult ShaderLoader::compile_glsl_to_spirv(
    const std::string& source,
    const std::string& filename,
    const ShaderCompileOptions& options)
{
    CompilationResult result;

    shaderc::Compiler compiler;
    shaderc::CompileOptions compile_options;

    // Set optimization level
    if (options.optimize) {
        compile_options.SetOptimizationLevel(shaderc_optimization_level_performance);
    } else {
        compile_options.SetOptimizationLevel(shaderc_optimization_level_zero);
    }

    // Enable debug info
    if (options.generate_debug_info) {
        compile_options.SetGenerateDebugInfo();
    }

    // Add preprocessor defines
    for (const auto& define : options.defines) {
        size_t eq_pos = define.find('=');
        if (eq_pos != std::string::npos) {
            compile_options.AddMacroDefinition(
                define.substr(0, eq_pos),
                define.substr(eq_pos + 1));
        } else {
            compile_options.AddMacroDefinition(define);
        }
    }

    // Set up include handler
    auto include_handler = std::make_shared<ShaderIncludeHandler>(options.include_dirs);
    compile_options.SetIncluder(std::make_unique<ShadercIncluder>(include_handler));

    // Target Vulkan 1.2 with SPIR-V 1.5
    compile_options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    compile_options.SetTargetSpirv(shaderc_spirv_version_1_5);

    // Compile
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
        source, shaderc_glsl_compute_shader, filename.c_str(), compile_options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        result.success = false;
        result.error_message = module.GetErrorMessage();
        return result;
    }

    result.success = true;
    result.spirv = std::vector<uint32_t>(module.cbegin(), module.cend());

    // Capture warnings
    if (module.GetNumWarnings() > 0) {
        result.warnings = module.GetErrorMessage(); // Contains warnings too
    }

    return result;
}

#else // !HAS_SHADERC

CompilationResult ShaderLoader::compile_glsl_to_spirv(
    const std::string& /*source*/,
    const std::string& filename,
    const ShaderCompileOptions& /*options*/)
{
    CompilationResult result;
    result.success = false;
    result.error_message = "Runtime shader compilation requires shaderc library. "
                          "Either link with shaderc or use precompiled .spv files. "
                          "Failed to compile: " + filename;
    return result;
}

#endif // HAS_SHADERC

bool ShaderLoader::compile_all_to_files(
    const std::filesystem::path& output_dir,
    const ShaderCompileOptions& options)
{
    std::filesystem::create_directories(output_dir);

    bool all_success = true;
    ShaderCompileOptions compile_opts = options;
    compile_opts.include_dirs.insert(compile_opts.include_dirs.begin(), shader_dir_);

    for (const auto& type : get_all_shader_types()) {
        const char* name = shader_type_to_name(type);
        std::filesystem::path comp_path = shader_dir_ / (std::string(name) + ".comp");
        std::filesystem::path spv_path = output_dir / (std::string(name) + ".spv");

        auto source = read_source_file(comp_path);
        if (!source) {
            std::cerr << "Failed to read: " << comp_path << "\n";
            all_success = false;
            continue;
        }

        auto result = compile_glsl_to_spirv(*source, comp_path.string(), compile_opts);
        if (!result.success) {
            std::cerr << "Failed to compile " << name << ": " << result.error_message << "\n";
            all_success = false;
            continue;
        }

        // Write SPIR-V to file
        std::ofstream out_file(spv_path, std::ios::binary);
        if (!out_file) {
            std::cerr << "Failed to write: " << spv_path << "\n";
            all_success = false;
            continue;
        }

        out_file.write(
            reinterpret_cast<const char*>(result.spirv.data()),
            result.spirv.size() * sizeof(uint32_t));

        std::cout << "Compiled: " << name << " -> " << spv_path << " ("
                  << result.spirv.size() * sizeof(uint32_t) << " bytes)\n";
    }

    return all_success;
}

} // namespace minecraft_vulkan
