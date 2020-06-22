#include "GPUMotionBlurUI.h"

#include "GPUMotionBlur.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_motion_blur_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_MOTION_BLUR;

    agn->name = "GPU Motion Blur";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_MOTION_BLUR, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&mb->frame_count));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&mb->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&mb->gmb_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = gpu_motion_blur_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_motion_blur_destroy;
    agn->externalise = gpu_motion_blur_externalise;
}

GPUMotionBlurFrame::GPUMotionBlurFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Motion Blur")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Frame Count"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_fc = new wxTextCtrl(panel, -1, wxT("4"));
    hbox_fc->Add(tc_fc, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_fc->GetValue();
    tc_fc->SetValue(wxT("4"));

    if (node_id == -1) {
        struct gpu_motion_blur* mb = new gpu_motion_blur();
        gpu_motion_blur_init(mb, stoi(str.c_str().AsChar()));

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_motion_blur_ui_graph_init(agn, (application_graph_component)mb, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;
        mb->frame_count = stoi(str.c_str().AsChar());
    }
    myApp->drawPane->Refresh();
}

void GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_fc->SetValue(wxT("4"));
}

void GPUMotionBlurFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;
        stringstream s_sw;
        s_sw << mb->frame_count;
        tc_fc->SetValue(wxString(s_sw.str()));
    }
    wxFrame::Show(true);
}