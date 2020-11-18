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
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&mb->calc_err));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->a));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->b));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->c));
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

    vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Frame Count"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);
    tc_fc = new wxTextCtrl(panel, -1, wxT("4"));
    hbox_fc->Add(tc_fc, 1);
    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_dt = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_dt = new wxStaticText(panel, -1, wxT("Blur Weight Distribution"));
    hbox_dt->Add(st_dt, 0, wxRight, 8);
    wxArrayString dt_choices;
    dt_choices.Add("Even");
    dt_choices.Add("Linear Roof");
    ch_dt = new wxChoice(panel, -1, wxDefaultPosition, wxDefaultSize, dt_choices);
    ch_dt->SetSelection(0);
    hbox_dt->Add(ch_dt, 1);
    ch_dt->Bind(wxEVT_COMMAND_CHOICE_SELECTED, &GPUMotionBlurFrame::OnWeightDistChange, this);
    vbox->Add(hbox_dt, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    hbox_wc = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_wc = new wxStaticText(panel, -1, wxT("Weight Center [0 - (Frame_Count-1)]"));
    hbox_wc->Add(st_wc, 0, wxRIGHT, 8);
    tc_wc = new wxTextCtrl(panel, -1, wxT("0.0"));
    hbox_wc->Add(tc_wc, 1);
    vbox->Add(hbox_wc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    hbox_c = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_c = new wxStaticText(panel, -1, wxT("Center Weight"));
    hbox_c->Add(st_c, 0, wxRIGHT, 8);
    tc_c = new wxTextCtrl(panel, -1, wxT("0.25"));
    hbox_c->Add(tc_c, 1);
    vbox->Add(hbox_c, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

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

void GPUMotionBlurFrame::OnWeightDistChange(wxCommandEvent& event) {
    int selection = ch_dt->GetSelection();
    if (selection == 0) {
        //Even
        hbox_wc->Show(false);
        hbox_wc->Layout();

        hbox_c->Show(false);
        hbox_c->Layout();
    } else if (selection == 1) {
        //Linear Roof
        hbox_wc->Show(true);
        hbox_wc->Layout();

        hbox_c->Show(true);
        hbox_c->Layout();
    }
    vbox->Layout();
}

void GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_fc->GetValue();
    tc_fc->SetValue(wxT("4"));

    int dt_type = ch_dt->GetSelection();
    ch_dt->SetSelection(0);

    float weight_center = stof(tc_wc->GetValue().c_str().AsChar());
    float center_weight = stof(tc_c->GetValue().c_str().AsChar());

    if (node_id == -1) {
        struct gpu_motion_blur* mb = new gpu_motion_blur();
        gpu_motion_blur_init(mb, stoi(str.c_str().AsChar()), dt_type, weight_center, center_weight);
        gpu_motion_blur_calculate_weights(mb);

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_motion_blur_ui_graph_init(agn, (application_graph_component)mb, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;
        mb->frame_count = stoi(str.c_str().AsChar());
        mb->weight_dist_type = dt_type;
        mb->frame_id_weight_center = weight_center;
        mb->c = center_weight;
        gpu_motion_blur_calculate_weights(mb);
    }
    
    myApp->drawPane->Refresh();
}

void GPUMotionBlurFrame::OnGPUMotionBlurFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_fc->SetValue(wxT("4"));
    ch_dt->SetSelection(0);
    tc_wc->SetValue(wxT("0"));
    tc_c->SetValue(wxT("0.25"));
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
        ch_dt->SetSelection(mb->weight_dist_type);

        stringstream s_wc;
        s_wc << mb->frame_id_weight_center;
        tc_wc->SetValue(wxString(s_wc.str()));

        stringstream s_c;
        s_c << mb->c;
        tc_c->SetValue(wxString(s_c.str()));
    }
    wxCommandEvent dummy;
    GPUMotionBlurFrame::OnWeightDistChange(dummy);
    wxFrame::Show(true);
}