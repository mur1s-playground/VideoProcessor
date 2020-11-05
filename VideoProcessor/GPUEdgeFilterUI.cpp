#include "GPUEdgeFilterUI.h"

#include "GPUEdgeFilter.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE = "Video Source";
const string TEXT_GPU_MEMORY_BUFFER_OUT = "GPU Memory Buffer Out";

void gpu_edge_filter_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_GPU_EDGE_FILTER;

    agn->name = "GPU Edge Filter";

    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_GPU_EDGE_FILTER, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct gpu_edge_filter* gd = (struct gpu_edge_filter*)agn->component;
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&gd->amplify));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&gd->vs_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&gd->gmb_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = gpu_edge_filter_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = gpu_edge_filter_destroy;
    agn->externalise = gpu_edge_filter_externalise;
}

GPUEdgeFilterFrame::GPUEdgeFilterFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("GPU Edge Filter")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_fc = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_fc = new wxStaticText(panel, -1, wxT("Amplify"));
    hbox_fc->Add(st_fc, 0, wxRIGHT, 8);

    tc_amplify = new wxTextCtrl(panel, -1, wxT("1.0"));
    hbox_fc->Add(tc_amplify, 1);

    vbox->Add(hbox_fc, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);


    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &GPUEdgeFilterFrame::OnGPUEdgeFilterButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &GPUEdgeFilterFrame::OnGPUEdgeFilterButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void GPUEdgeFilterFrame::OnGPUEdgeFilterButtonOk(wxCommandEvent& event) {
    this->Hide();
   
    float amplify = stof(string(tc_amplify->GetValue()));

    if (node_id == -1) {
        struct gpu_edge_filter* gef = new gpu_edge_filter();
        gpu_edge_filter_init(gef, amplify);

        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        gpu_edge_filter_ui_graph_init(agn, (application_graph_component)gef, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;

        gef->amplify = amplify;
    }
    myApp->drawPane->Refresh();
}

void GPUEdgeFilterFrame::OnGPUEdgeFilterButtonClose(wxCommandEvent& event) {
    wxString amplify_str;
    amplify_str << 1.0;
    tc_amplify->SetValue(amplify_str);

    this->Hide();
}

void GPUEdgeFilterFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;

        wxString amplify_str;
        amplify_str << gef->amplify;
        tc_amplify->SetValue(amplify_str);
    }
    wxFrame::Show(true);
}