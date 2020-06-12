#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void video_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class VideoSourceFrame : public wxFrame {
	
public:
	wxTextCtrl* tc;

	wxTextCtrl* tc_width;
	wxTextCtrl* tc_height;
	wxTextCtrl* tc_channels;
	wxChoice* ch_direction;

	VideoSourceFrame(wxWindow *parent);

	void OnVideoSourceFrameButtonOk(wxCommandEvent& event);
	void OnVideoSourceFrameButtonClose(wxCommandEvent& event);
};