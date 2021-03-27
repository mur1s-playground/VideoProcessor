#include "Statistics3DUI.h"
#include "Statistics3D.h"

#include "MainUI.h"

const string TEXT_MEMORY_BUFFER_SHARED_STATE = "SMB Shared State";
const string TEXT_VIDEO_SOURCE = "Video Source OUT";

void statistics_3d_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_STATISTICS_3D;

    agn->name = "Statistics 3D";
    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_CAMERA_CONTROL, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct statistics_3d* s3d = (struct statistics_3d*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_MEMORY_BUFFER_SHARED_STATE));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&s3d->smb_shared_state);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));


    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE));

    pair<enum application_graph_component_type, void*> inner_in0 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&s3d->vs_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in0));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = statistics_3d_loop;
    agn->process_run = false;

    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = statistics_3d_destroy;
    agn->externalise = statistics_3d_externalise;
}

Statistics3DFrame::Statistics3DFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Statistics 3D")) {
    node_graph_id = -1;
    node_id = -1;

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &Statistics3DFrame::OnStatistics3DFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &Statistics3DFrame::OnStatistics3DFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void Statistics3DFrame::OnStatistics3DFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    if (node_id == -1) {
        struct statistics_3d* s3d = new struct statistics_3d();
        statistics_3d_init(s3d);
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        statistics_3d_ui_graph_init(agn, (application_graph_component)s3d, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct statistics_3d* s3d = new struct statistics_3d();
    }
    myApp->drawPane->Refresh();
}

void Statistics3DFrame::OnStatistics3DFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
}

void Statistics3DFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct statistics_3d* cc = (struct statistics_3d*)agn->component;
    }
    wxFrame::Show(true);
}