project(sm-source)

if(MSVC)
	set(sm-source_PLATFORM_DEPS
		w32-pthreads)
endif()

set(sm-source_SOURCES
	sm-source.c)

if(WIN32)
	set(MODULE_DESCRIPTION "OBS shared memory module")
	configure_file(${CMAKE_SOURCE_DIR}/cmake/winrc/obs-module.rc.in sm-source.rc)
	list(APPEND sm-source_SOURCES
		sm-source.rc)
endif()

add_library(sm-source MODULE
	${sm-source_SOURCES})
target_link_libraries(sm-source
	libobs
	${sm-source_PLATFORM_DEPS})
set_target_properties(sm-source PROPERTIES FOLDER "plugins")

install_obs_plugin_with_data(sm-source data)
