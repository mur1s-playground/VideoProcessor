#pragma once

#include "ApplicationGraph.h"

void gpu_composer_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUComposerFrame : public wxFrame {

public:
	wxTextCtrl* tc;

	GPUComposerFrame(wxWindow* parent);

	void OnGPUComposerFrameButtonOk(wxCommandEvent& event);
	void OnGPUComposerFrameButtonClose(wxCommandEvent& event);
};
