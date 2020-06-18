#pragma once

#include "ApplicationGraph.h"

void gpu_composer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUComposerFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxTextCtrl* tc;

	wxButton* ok_button;

	GPUComposerFrame(wxWindow* parent);

	void OnGPUComposerFrameButtonOk(wxCommandEvent& event);
	void OnGPUComposerFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
