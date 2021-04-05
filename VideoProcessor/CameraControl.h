#pragma once

#include <windows.h>

#include <vector>
#include <string>
#include "Vector2.h"
#include "Vector3.h"

#include "SharedMemoryBuffer.h"
#include "VideoSource.h"
#include "Statistic.h"

using namespace std;

struct statistic_angle_denoiser {
	float				angle;
	float				angle_stability;

	float* angle_distribution;
	int					angle_distribution_size;
	int					angle_distribution_idx_latest;

	float* angle_distribution_weights;
};

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

struct cam_detection {
	int						class_id;
	float					score;

	int						x1, x2, y1, y2;
	unsigned long long		timestamp;
};

struct cam_detection_3d {
	int						class_id;
	float					score;

	vector3<float>			position;
	vector3<float>			velocity;

	vector3<float>			ray_position[5];
	vector3<float>			ray_direction[5];

	unsigned long long		timestamp;
};

enum cam_calibration_process_state {
	CCPS_CALIBRATION_OBJECT_SEARCH,
	CCPS_CALIBRATION_OBJECT_LOST,
	CCPS_CALIBRATION_OBJECT_DETECT_STABLE,

	/*
	CCPS_CALIBRATION_OBJECT_TOP_LEFT,
	CCPS_CALIBRATION_OBJECT_BOTTOM_RIGHT,
	*/

	/*
	//|-----------|
	//|>    >    v|
	//|^   (^)   v|
	//|^    <    <|
	//|-----------|
	CCPS_CALIBRATION_OBJECT_CENTER,
	CCPS_CALIBRATION_OBJECT_CENTER_TOP,
	CCPS_CALIBRATION_OBJECT_TOP_RIGHT,
	CCPS_CALIBRATION_OBJECT_CENTER_RIGHT,
	CCPS_CALIBRATION_OBJECT_BOTTOM_RIGHT,
	CCPS_CALIBRATION_OBJECT_CENTER_BOTTOM,
	CCPS_CALIBRATION_OBJECT_BOTTOM_LEFT,
	CCPS_CALIBRATION_OBJECT_CENTER_LEFT,
	CCPS_CALIBRATION_OBJECT_TOP_LEFT,
	*/	

	CCPS_CALIBRATION_OBJECT_GET_MATRIX,
	
	CCPS_CALIBRATION_DONE
};

struct cam_calibration_process {
	int tick;
	
	enum cam_calibration_process_state ccps;

	vector2<float>			camera_starting_angles;
	
	vector2<float>			calibration_object_found_angles;
	struct cam_detection	calibration_object_found;

	vector2<int>			calibration_discretization;

	vector2<float>			calibration_object_center_angles;
	struct cam_detection	calibration_object_center;

	int															calibration_object_a_cd_i;
	vector<pair<struct vector2<float>, struct cam_detection>>	calibration_object_a_cd;
	/*
	vector2<float>			calibration_object_center_angles;
	struct cam_detection	calibration_object_center;
	
	vector2<float>			calibration_object_center_top_angles;
	struct cam_detection	calibration_object_center_top;

	vector2<float>			calibration_object_center_bottom_angles;
	struct cam_detection	calibration_object_center_bottom;

	vector2<float>			calibration_object_center_left_angles;
	struct cam_detection	calibration_object_center_left;

	vector2<float>			calibration_object_center_right_angles;
	struct cam_detection	calibration_object_center_right;

	
	vector2<float>			calibration_object_top_left_angles;
	struct cam_detection	calibration_object_top_left;

	vector2<float>			calibration_object_bottom_right_angles;
	struct cam_detection	calibration_object_bottom_right;
	*/
	enum cam_calibration_process_state ccps_store;
};

struct cam_calibration {
	float					d_1;
	struct vector3<float>	position;
	
	struct vector2<int>		lens_quantization_size;
	struct statistic_unscatter_triangulation_2d *lens_north_pole;
	struct statistic_unscatter_triangulation_2d *lens_horizon;
	
	struct vector2<float>	lens_fov;
};

float cam_detection_get_center(struct cam_detection* cd, bool horizontal);
struct vector2<float> cam_detection_get_center(struct cam_detection* cd);

struct cam_detection_history {
	int							latest_idx;
	int							latest_count;
	struct cam_detection*		history;

	int							size;
};

struct cam_awareness {
	struct statistic_angle_denoiser		north_pole;
	struct statistic_angle_denoiser		horizon;

	struct cam_detection_history		detection_history;

	struct cam_calibration				calibration;

	struct vector2<float>				resolution_offset;
};

struct camera_control_shared_state {
	struct vector3<float>				position;
	struct vector2<float>				fov;

	float								np_angle;
	float								np_stability;
	float								np_sensor;

	float								horizon_angle;
	float								horizon_stability;
	float								horizon_sensor;
	
	int									latest_detections_used_ct;
	struct cam_detection				latest_detections[5];

	struct vector3<float>				latest_detections_rays_origin[15];
	struct vector3<float>				latest_detections_rays[15];
	struct vector3<float>				latest_detections_objects[5];

	struct cam_detection_3d				latest_detections_3d[5];
};

struct camera_control {
	int camera_count;

	string camera_meta_path;
	string sensors_path;
	string calibration_path;

	bool calibration;

	struct video_source* vs_cams;

	struct shared_memory_buffer* smb_det;

	struct cam_meta_data* cam_meta;

	struct cam_sensors* cam_sens;
	unsigned long long* cam_sens_timestamps;

	struct cam_awareness* cam_awareness;

	struct shared_memory_buffer* smb_shared_state;
	int							shared_state_size_req;

	struct statistic_heatmap *heatmap_general;
	struct statistic_vectorfield_3d* velocity_vectorfield_3d;

	struct shared_memory_buffer* smb_detection_sim;
};

void camera_control_init(struct camera_control* cc, int camera_count, string camera_meta_path, string sensors_path, string calibration_path);
void camera_control_on_input_connect(struct application_graph_node* agn, int input_id);

DWORD* camera_control_loop(LPVOID args);

void camera_control_externalise(struct application_graph_node* agn, string& out_str);
void camera_control_load(struct camera_control* cc, ifstream& in_f);
void camera_control_destroy(struct application_graph_node* agn);
