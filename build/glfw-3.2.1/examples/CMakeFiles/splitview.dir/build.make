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
include glfw-3.2.1/examples/CMakeFiles/splitview.dir/depend.make

# Include the progress variables for this target.
include glfw-3.2.1/examples/CMakeFiles/splitview.dir/progress.make

# Include the compile flags for this target's objects.
include glfw-3.2.1/examples/CMakeFiles/splitview.dir/flags.make

glfw-3.2.1/examples/splitview.app/Contents/Resources/glfw.icns: ../glfw-3.2.1/examples/glfw.icns
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Copying OS X content glfw-3.2.1/examples/splitview.app/Contents/Resources/glfw.icns"
	$(CMAKE_COMMAND) -E copy /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/examples/glfw.icns glfw-3.2.1/examples/splitview.app/Contents/Resources/glfw.icns

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o: glfw-3.2.1/examples/CMakeFiles/splitview.dir/flags.make
glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o: ../glfw-3.2.1/examples/splitview.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/musk/Desktop/graphix/RayTracer/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/splitview.dir/splitview.c.o   -c /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/examples/splitview.c

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/splitview.dir/splitview.c.i"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/examples/splitview.c > CMakeFiles/splitview.dir/splitview.c.i

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/splitview.dir/splitview.c.s"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/examples/splitview.c -o CMakeFiles/splitview.dir/splitview.c.s

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.requires:
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.requires

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.provides: glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.requires
	$(MAKE) -f glfw-3.2.1/examples/CMakeFiles/splitview.dir/build.make glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.provides.build
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.provides

glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.provides.build: glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o: glfw-3.2.1/examples/CMakeFiles/splitview.dir/flags.make
glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o: ../glfw-3.2.1/deps/glad.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/musk/Desktop/graphix/RayTracer/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/splitview.dir/__/deps/glad.c.o   -c /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/deps/glad.c

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/splitview.dir/__/deps/glad.c.i"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/deps/glad.c > CMakeFiles/splitview.dir/__/deps/glad.c.i

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/splitview.dir/__/deps/glad.c.s"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/deps/glad.c -o CMakeFiles/splitview.dir/__/deps/glad.c.s

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.requires:
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.requires

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.provides: glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.requires
	$(MAKE) -f glfw-3.2.1/examples/CMakeFiles/splitview.dir/build.make glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.provides.build
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.provides

glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.provides.build: glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o

# Object files for target splitview
splitview_OBJECTS = \
"CMakeFiles/splitview.dir/splitview.c.o" \
"CMakeFiles/splitview.dir/__/deps/glad.c.o"

# External object files for target splitview
splitview_EXTERNAL_OBJECTS =

glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview: glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o
glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview: glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o
glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview: glfw-3.2.1/examples/CMakeFiles/splitview.dir/build.make
glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview: glfw-3.2.1/src/libglfw3.a
glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview: glfw-3.2.1/examples/CMakeFiles/splitview.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C executable splitview.app/Contents/MacOS/splitview"
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/splitview.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
glfw-3.2.1/examples/CMakeFiles/splitview.dir/build: glfw-3.2.1/examples/splitview.app/Contents/MacOS/splitview
glfw-3.2.1/examples/CMakeFiles/splitview.dir/build: glfw-3.2.1/examples/splitview.app/Contents/Resources/glfw.icns
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/build

glfw-3.2.1/examples/CMakeFiles/splitview.dir/requires: glfw-3.2.1/examples/CMakeFiles/splitview.dir/splitview.c.o.requires
glfw-3.2.1/examples/CMakeFiles/splitview.dir/requires: glfw-3.2.1/examples/CMakeFiles/splitview.dir/__/deps/glad.c.o.requires
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/requires

glfw-3.2.1/examples/CMakeFiles/splitview.dir/clean:
	cd /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples && $(CMAKE_COMMAND) -P CMakeFiles/splitview.dir/cmake_clean.cmake
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/clean

glfw-3.2.1/examples/CMakeFiles/splitview.dir/depend:
	cd /Users/musk/Desktop/graphix/RayTracer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/musk/Desktop/graphix/RayTracer /Users/musk/Desktop/graphix/RayTracer/glfw-3.2.1/examples /Users/musk/Desktop/graphix/RayTracer/build /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples /Users/musk/Desktop/graphix/RayTracer/build/glfw-3.2.1/examples/CMakeFiles/splitview.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : glfw-3.2.1/examples/CMakeFiles/splitview.dir/depend

