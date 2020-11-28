#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

class ApplicationGraphNodeSettingsFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_tps_target;

	wxTextCtrl* tc_hotkey;

	ApplicationGraphNodeSettingsFrame(wxWindow* parent);

	void OnApplicationGraphNodeSettingsFrameButtonOk(wxCommandEvent& event);
	void OnApplicationGraphNodeSettingsFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};

