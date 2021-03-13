#pragma once

#include <windows.h>

#include <string>
#include "Vector2.h"

#include "SharedMemoryBuffer.h"
#include "VideoSource.h"
#include "Statistic.h"

using namespace std;

struct cam_meta_data {
	struct vector2<int> resolution;
};

struct cam_sensors {
	int compass;
	int tilt;
	bool tilt_min;
	bool tilt_max;
	bool zoom_min;
	bool zoom_max;
};

enum cam_engine {
	ENGINE_HORIZONTAL,
	ENGINE_VERTICAL,
	ENGINE_ZOOM,
};

struct cam_control {
	int id;
	enum cam_engine ngin;
	float volt_time;
	float volt[24];
	float volt_time_current;
};

struct cam_awareness {
	struct statistic_angle_denoiser		north_pole;
	struct statistic_angle_denoiser		horizon;

	struct vector2<int> lens_distortion_quantization_size;
	float*				lens_distortion_factor;

	struct vector2<int> lens_fov;
};

struct camera_control {
	int camera_count;

	string camera_meta_path;
	string sensors_path;

	bool calibration;

	struct video_source* vs_cams;

	struct shared_memory_buffer* smb_det;

	struct cam_meta_data* cam_meta;

	struct cam_sensors* cam_sens;
	unsigned long long* cam_sens_timestamps;

	struct cam_awareness* cam_awareness;
};

void camera_control_init(struct camera_control* cc, int camera_count, string camera_meta_path, string sensors_path);
void camera_control_on_input_connect(struct application_graph_node* agn, int input_id);

DWORD* camera_control_loop(LPVOID args);

void camera_control_externalise(struct application_graph_node* agn, string& out_str);
void camera_control_load(struct camera_control* cc, ifstream& in_f);
void camera_control_destroy(struct application_graph_node* agn);
