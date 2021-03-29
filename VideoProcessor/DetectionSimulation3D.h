#pragma once

#include "ApplicationGraph.h"

#include "Vector3.h"
#include <vector>
#include <map>

struct detection_simulation_3d {
	struct shared_memory_buffer		*smb_detections;
	int								smb_size_req;

	vector<struct vector3<float>>	camera_positions;

	vector<struct vector3<float>>	keypoints;
	map<int, std::vector<int>>		flowmap;

	float							probability_of_change;
	int								object_count;
	int								detected_class_id;

	vector<int>						keypoint_active;
	vector<int>						keypoint_next;

	int								tick_speed;
	
};

void detection_simulation_3d_init(struct detection_simulation_3d* ds3d);

DWORD* detection_simulation_3d_loop(LPVOID args);
void detection_simulation_3d_externalise(struct application_graph_node* agn, string& out_str);
void detection_simulation_3d_load(struct detection_simulation_3d* ds3d, ifstream& in_f);
void detection_simulation_3d_destroy(struct application_graph_node* agn);