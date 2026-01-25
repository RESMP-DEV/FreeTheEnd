"""Test shader compilation and syntax validation."""

import subprocess
from pathlib import Path

import pytest

SHADERS_DIR = Path(__file__).parent.parent / "cpp" / "shaders"


def get_all_shaders():
    """Get all shader files (both .comp and .glsl)."""
    return list(SHADERS_DIR.glob("*.comp")) + list(SHADERS_DIR.glob("*.glsl"))


def get_compilable_shaders():
    """Get only standalone compute shaders (.comp) that can be compiled.
    Excludes .glsl include files which are not standalone.
    """
    return list(SHADERS_DIR.glob("*.comp"))


@pytest.fixture
def shader_compiler():
    """Check if glslc (preferred) or glslangValidator is available."""
    # Prefer glslc as it supports #include directives
    result = subprocess.run(["which", "glslc"], capture_output=True, text=True)
    if result.returncode == 0:
        return "glslc"

    result = subprocess.run(["which", "glslangValidator"], capture_output=True, text=True)
    if result.returncode == 0:
        return "glslangValidator"

    pytest.skip("Neither glslc nor glslangValidator installed (brew install shaderc or glslang)")
    return None


class TestShaderSyntax:
    """Test shader files for basic syntax validity."""

    @pytest.mark.parametrize("shader_file", get_compilable_shaders(), ids=lambda p: p.name)
    def test_shader_parses(self, shader_file, shader_compiler):
        """Each shader should parse without syntax errors."""
        if shader_compiler == "glslc":
            # glslc supports #include directives natively
            cmd = [shader_compiler, "-c", str(shader_file), "-o", "/dev/null"]
        else:
            # glslangValidator needs explicit include support
            cmd = [shader_compiler, "-V", str(shader_file)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SHADERS_DIR),
        )
        assert result.returncode == 0, (
            f"Shader {shader_file.name} failed:\n{result.stdout}{result.stderr}"
        )

    def test_shaders_exist(self):
        """Verify we have shaders to test."""
        shaders = get_all_shaders()
        assert len(shaders) > 0, "No shader files found"
        print(f"Found {len(shaders)} shader files")


class TestShaderStructure:
    """Test shader file structure and conventions."""

    @pytest.mark.parametrize(
        "shader_file", [f for f in get_all_shaders() if f.suffix == ".comp"], ids=lambda p: p.name
    )
    def test_has_version_directive(self, shader_file):
        """Standalone compute shaders should have GLSL version directive."""
        content = shader_file.read_text()
        assert "#version" in content, f"{shader_file.name} missing #version directive"

    @pytest.mark.parametrize(
        "shader_file", [f for f in get_all_shaders() if f.suffix == ".comp"], ids=lambda p: p.name
    )
    def test_compute_has_local_size(self, shader_file):
        """Compute shaders should define local_size."""
        content = shader_file.read_text()
        assert "local_size" in content, f"{shader_file.name} missing local_size"

    @pytest.mark.parametrize(
        "shader_file", [f for f in get_all_shaders() if f.suffix == ".comp"], ids=lambda p: p.name
    )
    def test_compute_has_main(self, shader_file):
        """Compute shaders should have main function."""
        content = shader_file.read_text()
        assert "void main()" in content, f"{shader_file.name} missing main()"
