#pragma once

#include <vector>

#include "ApplicationGraph.h"

using namespace std;


extern vector<pair<enum application_graph_component_type, void*>> ui_manager_frame_store;

void ui_manager_show_frame(enum application_graph_component_type agct, int node_graph_id, int node_id = -1);