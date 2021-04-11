#include "CameraControlUI.h"
#include "CameraControl.h"

#include "MainUI.h"

const string TEXT_VIDEO_SOURCE_CAMERAS = "Video Source Cameras";
const string TEXT_MEMORY_BUFFER_DET = "SMB Detections";
const string TEXT_MEMORY_BUFFER_SHARED_STATE = "SMB Shared State";
const string TEXT_MEMORY_BUFFER_DETECTION_SIM = "SMB Detection Simulation";
const string TEXT_STATISTICS_3D_IN = "Statistics 3D In";

void camera_control_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_CAMERA_CONTROL;

    agn->name = "Camera Control";
    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_CAMERA_CONTROL, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct camera_control* cc = (struct camera_control*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_CAMERAS));

    pair<enum application_graph_component_type, void*> inner_in0 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&cc->vs_cams);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in0));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_MEMORY_BUFFER_DET));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&cc->smb_det);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_MEMORY_BUFFER_SHARED_STATE));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&cc->smb_shared_state);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&cc->shared_state_size_req));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_MEMORY_BUFFER_DETECTION_SIM));

    pair<enum application_graph_component_type, void*> inner_in3 = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&cc->smb_detection_sim);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in3));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_STATISTICS_3D_IN));

    pair<enum application_graph_component_type, void*> inner_in4 = pair<enum application_graph_component_type, void*>(AGCT_STATISTICS_3D, (void*)&cc->statistics_3d_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in4));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = camera_control_loop;
    agn->process_run = false;

    agn->on_input_connect = camera_control_on_input_connect;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = camera_control_destroy;
    agn->externalise = camera_control_externalise;
}

CameraControlFrame::CameraControlFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Camera Control")) {
    node_graph_id = -1;
    node_id = -1;

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);


    wxBoxSizer* hbox_c_count = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_count = new wxStaticText(panel, -1, wxT("Camera Count"));
    hbox_c_count->Add(st_count, 0, wxRIGHT, 8);

    tc_camera_count = new wxTextCtrl(panel, -1, wxT(""));
    hbox_c_count->Add(tc_camera_count, 1);

    vbox->Add(hbox_c_count, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_m_path = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_m_path = new wxStaticText(panel, -1, wxT("Camera Meta Path"));
    hbox_m_path->Add(st_m_path, 0, wxRIGHT, 8);

    tc_m_path = new wxTextCtrl(panel, -1, wxT(""));
    hbox_m_path->Add(tc_m_path, 1);

    vbox->Add(hbox_m_path, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_s_path = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_s_path = new wxStaticText(panel, -1, wxT("Sensors Path"));
    hbox_s_path->Add(st_s_path, 0, wxRIGHT, 8);

    tc_s_path = new wxTextCtrl(panel, -1, wxT(""));
    hbox_s_path->Add(tc_s_path, 1);

    vbox->Add(hbox_s_path, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_c_path = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_c_path = new wxStaticText(panel, -1, wxT("Calibration Path"));
    hbox_c_path->Add(st_c_path, 0, wxRIGHT, 8);

    tc_c_path = new wxTextCtrl(panel, -1, wxT(""));
    hbox_c_path->Add(tc_c_path, 1);

    vbox->Add(hbox_c_path, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_calibrate = new wxBoxSizer(wxHORIZONTAL);
    calibrate_button = new wxButton(panel, -1, wxT("Calibrate"), wxDefaultPosition, wxSize(70, 30));
    calibrate_button->Bind(wxEVT_BUTTON, &CameraControlFrame::OnCameraControlFrameButtonCalibrate, this);
    hbox_calibrate->Add(calibrate_button, 0);

    vbox->Add(hbox_calibrate, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &CameraControlFrame::OnCameraControlFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &CameraControlFrame::OnCameraControlFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void CameraControlFrame::OnCameraControlFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    if (node_id == -1) {

        int camera_count = stoi(tc_camera_count->GetValue().c_str().AsChar());
        tc_camera_count->SetValue("1");
    
        string camera_meta_path(tc_m_path->GetValue().c_str().AsChar());
        string camera_sensors_path(tc_s_path->GetValue().c_str().AsChar());
        string camera_calibration_path(tc_c_path->GetValue().c_str().AsChar());

        struct camera_control* cc = new struct camera_control();
        camera_control_init(cc, camera_count, camera_meta_path, camera_sensors_path, camera_calibration_path);
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        camera_control_ui_graph_init(agn, (application_graph_component)cc, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct camera_control* cc = (struct camera_control*)agn->component;
    }
    myApp->drawPane->Refresh();
}

void CameraControlFrame::OnCameraControlFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_camera_count->SetValue("1");
    tc_m_path->SetValue("");
    tc_s_path->SetValue("");
    tc_c_path->SetValue("");
}

void CameraControlFrame::OnCameraControlFrameButtonCalibrate(wxCommandEvent& event) {
    if (node_id != -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct camera_control* cc = (struct camera_control*)agn->component;
        cc->calibration = true;
    }
}

void CameraControlFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct camera_control* cc = (struct camera_control*)agn->component;

        stringstream s_cc;
        s_cc << cc->camera_count;
        tc_camera_count->SetValue(wxString(s_cc.str()));

        tc_m_path->SetValue(wxString(cc->camera_meta_path));
        tc_s_path->SetValue(wxString(cc->sensors_path));
        tc_c_path->SetValue(wxString(cc->calibration_path));
    }
    wxFrame::Show(true);
}