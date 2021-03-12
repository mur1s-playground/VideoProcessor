#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include <vector>
#include <map>

#include "Clock.h"

using namespace std;

struct application_graph_tps_balancer {
	int tps_current;
	int tps_target;

	timespec start;
	timespec stop;

	int overlap;
	int sleep_ms;

	bool started;
};

enum application_graph_component_type {
	AGCT_NONE,
	AGCT_ANY_NODE_SETTINGS,
	AGCT_IM_SHOW,
	AGCT_VIDEO_SOURCE,
	AGCT_SHARED_MEMORY_BUFFER,
	AGCT_GPU_MEMORY_BUFFER,
	AGCT_GPU_DENOISE,
	AGCT_MASK_RCNN,
	AGCT_GPU_VIDEO_ALPHA_MERGE,
	AGCT_GPU_COMPOSER,
	AGCT_GPU_COMPOSER_ELEMENT,
	AGCT_GPU_MOTION_BLUR,
	AGCT_GPU_GAUSSIAN_BLUR,
	AGCT_GPU_EDGE_FILTER,
	AGCT_GPU_PALETTE_FILTER,
	AGCT_GPU_AUDIOVISUAL,
	AGCT_AUDIO_SOURCE,
	AGCT_MINI_GINE,
	AGCT_GPU_GREEN_SCREEN,
	AGCT_CAMERA_CONTROL
};

#define application_graph_component void *

enum application_graph_node_vtype {
	AGNVT_SEPARATOR,
	AGNVT_STRING,
	AGNVT_STRING_LIST,
	AGNVT_INT,
	AGNVT_FLOAT,
	AGNVT_BOOL,
	AGNVT_UCHAR
};

struct application_graph_node {
	int n_id;

	application_graph_component component;
	enum application_graph_component_type component_type;

	string name;

	int pos_x;
	int pos_y;
	vector<int> heights;
	int max_width;

	vector<pair<enum application_graph_node_vtype, void*>> v;

	vector<pair<int, pair<enum application_graph_component_type, void*>>> inputs;
	vector<pair<int, pair<enum application_graph_component_type, void*>>> outputs;

	void *process;
	bool process_run;
	struct application_graph_tps_balancer process_tps_balancer;
	int	start_stop_hotkey;

	void *on_input_connect;
	void *on_input_edit;
	void *on_input_disconnect;

	void* on_delete;

	void *externalise;
};

struct application_graph_edge {
	pair<application_graph_node*, int> from;
	pair<application_graph_node*, int> to;
};

struct application_graph {
	string									name;
	vector<struct application_graph_node *> nodes;
	vector<struct application_graph_edge *> edges;
};

unsigned long long application_graph_tps_balancer_get_time();

void application_graph_tps_balancer_init(struct application_graph_node* agn, int tps_target);
void application_graph_tps_balancer_timer_start(struct application_graph_node* agn);
void application_graph_tps_balancer_timer_stop(struct application_graph_node* agn);
int application_graph_tps_balancer_get_sleep_ms(struct application_graph_node* agn);
void application_graph_tps_balancer_sleep(struct application_graph_node* agn);

int application_graph_is_on_input(int id, int node_id, int pos_x, int pos_y, float* dist_out);
int application_graph_is_on_output(int id, int node_id, int pos_x, int pos_y, float* dist_out);

void application_graph_add_edge(int id, int pos_x1, int pos_y1, int pos_x2, int pos_y2);

void application_graph_start_stop_node(int id, int node_id);
//void application_graph_start_node(int id, int pos_x, int pos_y);
void application_graph_start_node(int id, int node_id);
void application_graph_stop_node(int id, int node_id);

void application_graph_draw_nodes(struct application_graph* ag, wxDC& dc);
void application_graph_draw_edges(struct application_graph* ag, wxDC& dc);

void application_graph_hovering_node(int application_graph_id);

void application_graph_delete_edge(int application_graph_id, int node_id, int input_id);
void application_graph_delete_edge(int application_graph_id, int edge_id, bool refresh = true);
void application_graph_delete_node(int application_graph_id, int node_id);

void application_graph_save(string base_dir, string name);
void application_graph_load(string base_dir, string name);

extern std::vector <struct application_graph*>		ags;

extern int											application_graph_active_id;
extern int											application_graph_hovering_node_id;