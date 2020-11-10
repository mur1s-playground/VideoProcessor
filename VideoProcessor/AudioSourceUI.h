#pragma once

#include "wx/wx.h"
#include "wx/sizer.h"

#include "ApplicationGraph.h"

void audio_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y);

class AudioSourceFrame : public wxFrame {
	int node_graph_id;
	int node_id;

public:
	wxTextCtrl* tc_device_id;
	wxTextCtrl* tc_channels;
	wxTextCtrl* tc_samples_per_sec;
	wxTextCtrl* tc_bits_per_sample;
	wxTextCtrl* tc_copy_to_gmb;


	AudioSourceFrame(wxWindow* parent);

	void OnAudioSourceFrameButtonOk(wxCommandEvent& event);
	void OnAudioSourceFrameButtonClose(wxCommandEvent& event);

	void Show(int node_graph_id, int node_id);
};