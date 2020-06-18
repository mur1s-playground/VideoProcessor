#include "GPUVideoAlphaMergeUI.h"

#include "GPUVideoAlphaMerge.h"

#include "MainUI.h"

const string TEXT_VIDEO_RGB = "Video Source RGB (GPU)";
const string TEXT_VIDEO_ALPHA = "Video Source Alpha (GPU)";

const string TEXT_VIDEO_OUT = "Video Source Output (GPU)";

void gpu_video_alpha_merge_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_VIDEO_ALPHA_MERGE;

    agn->name = "GPU Video Alpha Merge";

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agc;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_RGB));
    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&vam->vs_rgb);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&vam->rgb_delay));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_ALPHA));
    pair<enum application_graph_component_type, void*> inner_in_2 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&vam->vs_alpha);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in_2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&vam->channel_id));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_OUT));
    pair<enum application_graph_component_type, void*> inner_in_3 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&vam->vs_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in_3));

    agn->process = gpu_video_alpha_merge_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_video_alpha_merge_destroy;
    agn->externalise = gpu_video_alpha_merge_externalise;
}

GPUVideoAlphaMergeFrame::GPUVideoAlphaMergeFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Video Alpha Merge"), wxPoint(50, 50), wxSize(260, 200)) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_delay = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_delay = new wxStaticText(panel, -1, wxT("RGB Delay"));
    hbox_delay->Add(st_delay, 0, wxRIGHT, 8);

    tc_d = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_delay->Add(tc_d, 1);

    vbox->Add(hbox_delay, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_channel = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_channel = new wxStaticText(panel, -1, wxT("Alpha Channel Id"));
    hbox_channel->Add(st_channel, 0, wxRIGHT, 8);

    tc = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_channel->Add(tc, 1);

    vbox->Add(hbox_channel, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUVideoAlphaMergeFrame::OnGPUVideoAlphaMergeFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUVideoAlphaMergeFrame::OnGPUVideoAlphaMergeFrameButtonClose, this);

    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUVideoAlphaMergeFrame::OnGPUVideoAlphaMergeFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str_d = tc_d->GetValue();
    tc_d->SetValue(wxT("0"));

    wxString str = tc->GetValue();
    tc->SetValue(wxT("0"));
    if (node_id == -1) {
        struct gpu_video_alpha_merge* vam = new gpu_video_alpha_merge();
        gpu_video_alpha_merge_init(vam, stoi(str_d.c_str().AsChar()), stoi(str.c_str().AsChar()));
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_video_alpha_merge_ui_graph_init(agn, (application_graph_component)vam, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;
        vam->channel_id = stoi(str.c_str().AsChar());
        vam->rgb_delay = stoi(str_d.c_str().AsChar());
    }
    myApp->drawPane->Refresh();
}

void GPUVideoAlphaMergeFrame::OnGPUVideoAlphaMergeFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc->SetValue(wxT("0"));
}

void GPUVideoAlphaMergeFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;

        stringstream s_d;
        s_d << vam->rgb_delay;
        tc_d->SetValue(wxString(s_d.str()));

        stringstream s_id;
        s_id << vam->channel_id;
        tc->SetValue(wxString(s_id.str()));
    }
    wxFrame::Show(true);
}