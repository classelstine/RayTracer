# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.0.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.0.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/musk/Desktop/graphix/RayTracer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/musk/Desktop/graphix/RayTracer/build

# Include any dependencies generated for this target.
include src/CMakeFiles/raytrace.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/raytrace.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/raytrace.dir/flags.make

src/CMakeFiles/raytrace.dir/raytrace.cpp.o: src/CMakeFiles/raytrace.dir/flags.make
src/CMakeFiles/raytrace.dir/raytrace.cpp.o: ../src/raytrace.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/musk/Desktop/graphix/RayTracer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/raytrace.dir/raytrace.cpp.o"
	cd /Users/musk/Desktop/graphix/RayTracer/build/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/raytrace.dir/raytrace.cpp.o -c /Users/musk/Desktop/graphix/RayTracer/src/raytrace.cpp

src/CMakeFiles/raytrace.dir/raytrace.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/raytrace.dir/raytrace.cpp.i"
	cd /Users/musk/Desktop/graphix/RayTracer/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/musk/Desktop/graphix/RayTracer/src/raytrace.cpp > CMakeFiles/raytrace.dir/raytrace.cpp.i

src/CMakeFiles/raytrace.dir/raytrace.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/raytrace.dir/raytrace.cpp.s"
	cd /Users/musk/Desktop/graphix/RayTracer/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/musk/Desktop/graphix/RayTracer/src/raytrace.cpp -o CMakeFiles/raytrace.dir/raytrace.cpp.s

src/CMakeFiles/raytrace.dir/raytrace.cpp.o.requires:
.PHONY : src/CMakeFiles/raytrace.dir/raytrace.cpp.o.requires

src/CMakeFiles/raytrace.dir/raytrace.cpp.o.provides: src/CMakeFiles/raytrace.dir/raytrace.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/raytrace.dir/build.make src/CMakeFiles/raytrace.dir/raytrace.cpp.o.provides.build
.PHONY : src/CMakeFiles/raytrace.dir/raytrace.cpp.o.provides

src/CMakeFiles/raytrace.dir/raytrace.cpp.o.provides.build: src/CMakeFiles/raytrace.dir/raytrace.cpp.o

# Object files for target raytrace
raytrace_OBJECTS = \
"CMakeFiles/raytrace.dir/raytrace.cpp.o"

# External object files for target raytrace
raytrace_EXTERNAL_OBJECTS =

raytrace: src/CMakeFiles/raytrace.dir/raytrace.cpp.o
raytrace: src/CMakeFiles/raytrace.dir/build.make
raytrace: glew/libglew.a
raytrace: glew/libglew.a
raytrace: glfw-3.2.1/src/libglfw3.a
raytrace: src/CMakeFiles/raytrace.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../raytrace"
	cd /Users/musk/Desktop/graphix/RayTracer/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/raytrace.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/raytrace.dir/build: raytrace
.PHONY : src/CMakeFiles/raytrace.dir/build

src/CMakeFiles/raytrace.dir/requires: src/CMakeFiles/raytrace.dir/raytrace.cpp.o.requires
.PHONY : src/CMakeFiles/raytrace.dir/requires

src/CMakeFiles/raytrace.dir/clean:
	cd /Users/musk/Desktop/graphix/RayTracer/build/src && $(CMAKE_COMMAND) -P CMakeFiles/raytrace.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/raytrace.dir/clean

src/CMakeFiles/raytrace.dir/depend:
	cd /Users/musk/Desktop/graphix/RayTracer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/musk/Desktop/graphix/RayTracer /Users/musk/Desktop/graphix/RayTracer/src /Users/musk/Desktop/graphix/RayTracer/build /Users/musk/Desktop/graphix/RayTracer/build/src /Users/musk/Desktop/graphix/RayTracer/build/src/CMakeFiles/raytrace.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/raytrace.dir/depend

