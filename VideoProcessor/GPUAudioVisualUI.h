#pragma once

#include "ApplicationGraph.h"

void gpu_audiovisual_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class GPUAudioVisualFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_name;
	wxTextCtrl* tc_dft_size;
	wxTextCtrl* tc_base_a;
	wxTextCtrl* tc_base_c;

	wxTextCtrl* tc_amplify;
	wxTextCtrl* tc_frame_names;

	wxChoice*	ch_active_theme;

	wxChoice*	ch_transition_theme;
	wxTextCtrl* tc_transition_fade;

	GPUAudioVisualFrame(wxWindow* parent);

	void OnGPUAudioVisualFrameButtonOk(wxCommandEvent& event);
	void OnGPUAudioVisualFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};