#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include <vector>

using namespace std;

enum application_graph_component_type {
	AGCT_NONE,
	AGCT_IM_SHOW,
	AGCT_VIDEO_SOURCE,
	AGCT_SHARED_MEMORY_BUFFER,
	AGCT_GPU_MEMORY_BUFFER,
	AGCT_GPU_DENOISE,
	AGCT_MASK_RCNN,
	AGCT_GPU_VIDEO_ALPHA_MERGE,
	AGCT_GPU_COMPOSER,
	AGCT_GPU_COMPOSER_ELEMENT
};

#define application_graph_component void *

enum application_graph_node_vtype {
	AGNVT_SEPARATOR,
	AGNVT_STRING,
	AGNVT_STRING_LIST,
	AGNVT_INT,
	AGNVT_FLOAT,
	AGNVT_BOOL
};

struct application_graph_node {
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

	void *on_input_connect;
};

struct application_graph_edge {
	pair<application_graph_node*, int> from;
	pair<application_graph_node*, int> to;
};

struct application_graph {
	vector<struct application_graph_node *> nodes;
	vector<struct application_graph_edge *> edges;
};

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


extern std::vector <struct application_graph*>   ags;
extern int application_graph_hovering_node_id;