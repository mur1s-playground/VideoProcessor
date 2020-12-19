#pragma once

#include "ApplicationGraph.h"

void mini_gine_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class MiniGineFrame : public wxFrame {
	int node_graph_id;
	int node_id;
public:
	wxTextCtrl* tc;

	wxTextCtrl* tc_entity_settings;

	wxTextCtrl* tc_entity_group_settings;

	wxButton* ok_button;

	MiniGineFrame(wxWindow* parent);

	void OnMiniGineFrameButtonApplyEntitySettings(wxCommandEvent& event);
	void OnMiniGineFrameButtonApplyEntityGroupSettings(wxCommandEvent& event);

	void OnMiniGineFrameButtonOk(wxCommandEvent& event);
	void OnMiniGineFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};
