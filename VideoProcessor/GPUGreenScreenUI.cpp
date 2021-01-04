#include "GPUGreenScreenUI.h"

#include "GPUGreenScreen.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_green_screen_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_GREEN_SCREEN;

    agn->name = "GPU Green Screen";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_GREEN_SCREEN, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_UCHAR, (void*)&mb->rgb[0]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_UCHAR, (void*)&mb->rgb[1]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_UCHAR, (void*)&mb->rgb[2]));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mb->threshold));
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

    agn->process = gpu_green_screen_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_green_screen_destroy;
    agn->externalise = gpu_green_screen_externalise;
}

GPUGreenScreenFrame::GPUGreenScreenFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Green Screen")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("R"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_r = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_fc->Add(tc_r, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_g = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_g = new wxStaticText(panel, -1, wxT("G"));
    hbox_g->Add(st_g, 0, wxRIGHT, 8);

    tc_g = new wxTextCtrl(panel, -1, wxT("255"));
    hbox_g->Add(tc_g, 1);

    vbox->Add(hbox_g, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_b = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_b = new wxStaticText(panel, -1, wxT("B"));
    hbox_b->Add(st_b, 0, wxRIGHT, 8);

    tc_b = new wxTextCtrl(panel, -1, wxT("0"));
    hbox_b->Add(tc_b, 1);

    vbox->Add(hbox_b, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_t = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_t = new wxStaticText(panel, -1, wxT("threshold"));
    hbox_t->Add(st_t, 0, wxRIGHT, 8);

    tc_threshold = new wxTextCtrl(panel, -1, wxT("50"));
    hbox_t->Add(tc_threshold, 1);

    vbox->Add(hbox_t, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUGreenScreenFrame::OnGPUGreenScreenFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUGreenScreenFrame::OnGPUGreenScreenFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUGreenScreenFrame::OnGPUGreenScreenFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc_r->GetValue();
    tc_r->SetValue(wxT("0"));

    wxString str_g = tc_g->GetValue();
    tc_g->SetValue(wxT("255"));

    wxString str_b = tc_b->GetValue();
    tc_b->SetValue(wxT("0"));

    wxString str_c = tc_threshold->GetValue();
    tc_threshold->SetValue(wxT("50"));

    if (node_id == -1) {
        vector3<unsigned char> rgb(stoi(str.c_str().AsChar()), stoi(str_g.c_str().AsChar()), stoi(str_b.c_str().AsChar()));

        struct gpu_green_screen* mb = new gpu_green_screen();
        gpu_green_screen_init(mb, rgb, stof(str_c.c_str().AsChar()));

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_green_screen_ui_graph_init(agn, (application_graph_component)mb, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;
        mb->rgb[0] = stoi(str.c_str().AsChar());
        mb->rgb[1] = stoi(str_g.c_str().AsChar());
        mb->rgb[2] = stoi(str_b.c_str().AsChar());
        mb->threshold = stof(str_c.c_str().AsChar());
    }
    myApp->drawPane->Refresh();
}

void GPUGreenScreenFrame::OnGPUGreenScreenFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_r->SetValue(wxT("0"));
    tc_g->SetValue(wxT("255"));
    tc_b->SetValue(wxT("0"));
    tc_threshold->SetValue(wxT("50"));
}

void GPUGreenScreenFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;
        
        stringstream s_sw;
        s_sw << mb->threshold;
        tc_threshold->SetValue(wxString(s_sw.str()));

        stringstream s_a;
        s_a << (int)mb->rgb[0];
        tc_r->SetValue(wxString(s_a.str()));

        stringstream s_g;
        s_g << (int)mb->rgb[1];
        tc_g->SetValue(wxString(s_g.str()));

        stringstream s_b;
        s_b << (int)mb->rgb[2];
        tc_b->SetValue(wxString(s_b.str()));
    }
    wxFrame::Show(true);
}