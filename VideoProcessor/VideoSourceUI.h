#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void video_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class VideoSourceFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxBoxSizer* vbox;

	wxChoice* ch_source_type;

	wxBoxSizer* hbox_devices;
	wxChoice* ch_devices;
	wxArrayString devices_choices;

	wxBoxSizer* hbox_path;
	wxTextCtrl* tc;

	wxBoxSizer* hbox_loop;
	wxChoice* ch_loop;

	wxBoxSizer* hbox_width;
	wxTextCtrl* tc_width;

	wxBoxSizer* hbox_height;
	wxTextCtrl* tc_height;

	wxBoxSizer* hbox_channels;
	wxTextCtrl* tc_channels;

	wxChoice* ch_direction;

	VideoSourceFrame(wxWindow *parent);

	void OnSourceTypeChange(wxCommandEvent& event);

	void OnVideoSourceFrameButtonOk(wxCommandEvent& event);
	void OnVideoSourceFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};