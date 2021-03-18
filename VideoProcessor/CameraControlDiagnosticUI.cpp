#include "CameraControlDiagnosticUI.h"
#include "CameraControlDiagnostic.h"

#include "MainUI.h"

const string TEXT_CAMERA_CONTROL = "Camera Control";
const string TEXT_VIDEO_SOURCE_OUT = "Video Source OUT";

void camera_control_diagnostic_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_CAMERA_CONTROL_DIAGNOSTIC;

    agn->name = "Camera Control Diagnostic";

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agc;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_CAMERA_CONTROL));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_CAMERA_CONTROL, (void*)&ccd->cc);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&ccd->vs_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = camera_control_diagnostic_loop;
    agn->process_run = false;
    agn->on_input_connect = camera_control_diagnostic_on_input_connect;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->externalise = camera_control_diagnostic_externalise;
    agn->on_delete = camera_control_diagnostic_destroy;
}

CameraControlDiagnosticFrame::CameraControlDiagnosticFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Camera Control Diagnostic")) {
    node_graph_id = -1;
    node_id = -1;

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);


    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &CameraControlDiagnosticFrame::OnCameraControlDiagnosticFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &CameraControlDiagnosticFrame::OnCameraControlDiagnosticFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void CameraControlDiagnosticFrame::OnCameraControlDiagnosticFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    if (node_id == -1) {
        struct camera_control_diagnostic* ccd = new camera_control_diagnostic();
        camera_control_diagnostic_init(ccd);
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        camera_control_diagnostic_ui_graph_init(agn, (application_graph_component)ccd, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    }
    else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct camera_control_diagnostic* ccd = (struct camera_control_diagnostic*)agn->component;
    }
    myApp->drawPane->Refresh();
}

void CameraControlDiagnosticFrame::OnCameraControlDiagnosticFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
}

void CameraControlDiagnosticFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct camera_control_diagnostic* gc = (struct camera_control_diagnostic*)agn->component;
    }
    wxFrame::Show(true);
}