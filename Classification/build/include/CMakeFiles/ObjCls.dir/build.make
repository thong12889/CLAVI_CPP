# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build

# Include any dependencies generated for this target.
include include/CMakeFiles/ObjCls.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include include/CMakeFiles/ObjCls.dir/compiler_depend.make

# Include the progress variables for this target.
include include/CMakeFiles/ObjCls.dir/progress.make

# Include the compile flags for this target's objects.
include include/CMakeFiles/ObjCls.dir/flags.make

include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o: include/CMakeFiles/ObjCls.dir/flags.make
include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o: /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/include/ObjectClassification.cpp
include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o: include/CMakeFiles/ObjCls.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o"
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o -MF CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o.d -o CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o -c /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/include/ObjectClassification.cpp

include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ObjCls.dir/ObjectClassification.cpp.i"
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/include/ObjectClassification.cpp > CMakeFiles/ObjCls.dir/ObjectClassification.cpp.i

include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ObjCls.dir/ObjectClassification.cpp.s"
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/include/ObjectClassification.cpp -o CMakeFiles/ObjCls.dir/ObjectClassification.cpp.s

# Object files for target ObjCls
ObjCls_OBJECTS = \
"CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o"

# External object files for target ObjCls
ObjCls_EXTERNAL_OBJECTS =

include/libObjCls.a: include/CMakeFiles/ObjCls.dir/ObjectClassification.cpp.o
include/libObjCls.a: include/CMakeFiles/ObjCls.dir/build.make
include/libObjCls.a: include/CMakeFiles/ObjCls.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libObjCls.a"
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && $(CMAKE_COMMAND) -P CMakeFiles/ObjCls.dir/cmake_clean_target.cmake
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ObjCls.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
include/CMakeFiles/ObjCls.dir/build: include/libObjCls.a
.PHONY : include/CMakeFiles/ObjCls.dir/build

include/CMakeFiles/ObjCls.dir/clean:
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include && $(CMAKE_COMMAND) -P CMakeFiles/ObjCls.dir/cmake_clean.cmake
.PHONY : include/CMakeFiles/ObjCls.dir/clean

include/CMakeFiles/ObjCls.dir/depend:
	cd /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/include /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include /home/ryowa/ryowa/nreal/ai/CLAVI_CPP/Classification/build/include/CMakeFiles/ObjCls.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : include/CMakeFiles/ObjCls.dir/depend

