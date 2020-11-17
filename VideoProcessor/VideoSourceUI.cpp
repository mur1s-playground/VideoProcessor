#include "VideoSourceUI.h"

#include <sstream>
#include <vector>

#include "VideoSource.h"

#include "Logger.h"

#include "MainUI.h"
#include "wx/wx.h"

using namespace std;

const string TEXT_MEMORY_BUFFER = "Memory Buffer";
const string TEXT_GPU_MEMORY_BUFFER = "GPU Memory Buffer";

void video_source_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_VIDEO_SOURCE;

    agn->name = "Video Source";
    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void *)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct video_source* vs = (struct video_source*)agn->component;
   
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&vs->name));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void *)&vs->video_width));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&vs->video_height));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&vs->video_channels));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&vs->is_open));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_BOOL, (void*)&vs->direction_smb_to_gmb));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_MEMORY_BUFFER));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&vs->smb);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size()-1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_GPU_MEMORY_BUFFER));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_GPU_MEMORY_BUFFER, (void*)&vs->gmb);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&vs->smb_size_req));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = video_source_loop;
    agn->process_run = false;

    agn->on_input_connect = video_source_on_input_connect;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = video_source_destory;
    agn->externalise = video_source_externalise;
}

VideoSourceFrame::VideoSourceFrame(wxWindow *parent) : wxFrame(parent, -1, wxT("Video Source")) {

    wxPanel* panel = new wxPanel(this, -1);

    vbox = new wxBoxSizer(wxVERTICAL);

    //Video Source Type
    wxBoxSizer* hbox_type = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_type = new wxStaticText(panel, -1, wxT("Video Source Type"));
    hbox_type->Add(st_type, 0, wxRight, 8);
    wxArrayString type_choices;
    type_choices.Add("Device");
    type_choices.Add("Path");
    type_choices.Add("Dummy");
    type_choices.Add("Desktop");
    ch_source_type = new wxChoice(panel, -1, wxDefaultPosition, wxDefaultSize, type_choices);
    ch_source_type->SetSelection(0);
    hbox_type->Add(ch_source_type, 1);
    ch_source_type->Bind(wxEVT_COMMAND_CHOICE_SELECTED, &VideoSourceFrame::OnSourceTypeChange, this);
    vbox->Add(hbox_type, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    //Devices
    hbox_devices = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_devices = new wxStaticText(panel, -1, wxT("Devices"));
    hbox_devices->Add(st_devices, 0, wxRight, 8);
    wxArrayString devices_choices;
    devices_choices.Add("dev 0");
    ch_devices = new wxChoice(panel, -1, wxDefaultPosition, wxDefaultSize, devices_choices);
    ch_devices->SetSelection(0);
    hbox_devices->Add(ch_devices, 1);
    vbox->Add(hbox_devices, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    //Path
    hbox_path = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_path = new wxStaticText(panel, -1, wxT("DeviceId/Path"));
    hbox_path->Add(st_path, 0, wxRIGHT, 8);
    tc = new wxTextCtrl(panel, -1, wxT(""));
    hbox_path->Add(tc, 1);
    hbox_path->Show(false);
    hbox_path->Layout();
    vbox->Add(hbox_path, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
    
    //Width
    hbox_width = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_width = new wxStaticText(panel, -1, wxT("Width"));
    hbox_width->Add(st_width, 0, wxRIGHT, 8);
    tc_width = new wxTextCtrl(panel, -1, wxT(""));
    hbox_width->Add(tc_width, 1);
    vbox->Add(hbox_width, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    //Height
    hbox_height = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_height = new wxStaticText(panel, -1, wxT("Height"));
    hbox_height->Add(st_height, 0, wxRIGHT, 8);
    tc_height = new wxTextCtrl(panel, -1, wxT(""));
    hbox_height->Add(tc_height, 1);
    vbox->Add(hbox_height, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    //Channels
    hbox_channels = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_channels = new wxStaticText(panel, -1, wxT("Channels"));
    hbox_channels->Add(st_channels, 0, wxRIGHT, 8);
    tc_channels = new wxTextCtrl(panel, -1, wxT(""));
    hbox_channels->Add(tc_channels, 1);
    vbox->Add(hbox_channels, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    //Copying direction
    wxBoxSizer* hbox_direction = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_direction = new wxStaticText(panel, -1, wxT("Copy Buffer?"));
    hbox_direction->Add(st_direction, 0, wxRIGHT, 8);
    wxArrayString choices;
    choices.Add("no");
    choices.Add("Host->GPU");
    choices.Add("GPU->Host");
    ch_direction = new wxChoice(panel, -1, wxDefaultPosition, wxDefaultSize, choices);
    ch_direction->SetSelection(0);
    hbox_direction->Add(ch_direction, 1);
    vbox->Add(hbox_direction, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    //Buttons
    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);
    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &VideoSourceFrame::OnVideoSourceFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &VideoSourceFrame::OnVideoSourceFrameButtonClose, this);  
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void VideoSourceFrame::OnSourceTypeChange(wxCommandEvent& event) {
    int selection = ch_source_type->GetSelection();
    if (selection == 0) {
        //Device
        hbox_devices->Show(true);
        hbox_devices->Layout();
        
        hbox_path->Show(false);
        hbox_path->Layout();

        hbox_width->Show(true);
        hbox_width->Layout();
        
        hbox_height->Show(true);
        hbox_height->Layout();
        
        hbox_channels->Show(true);
        hbox_channels->Layout();
    } else if (selection == 1) {
        //Path
        hbox_devices->Show(false);
        hbox_devices->Layout();

        hbox_path->Show(true);
        hbox_path->Layout();

        hbox_width->Show(false);
        hbox_width->Layout();

        hbox_height->Show(false);
        hbox_height->Layout();

        hbox_channels->Show(false);
        hbox_channels->Layout();
    } else if (selection == 2) {
        //Dummy
        hbox_devices->Show(false);
        hbox_devices->Layout();

        hbox_path->Show(false);
        hbox_path->Layout();

        hbox_width->Show(true);
        hbox_width->Layout();

        hbox_height->Show(true);
        hbox_height->Layout();

        hbox_channels->Show(true);
        hbox_channels->Layout();
    } else if (selection == 3) {
        //Desktop
        hbox_devices->Show(false);
        hbox_devices->Layout();

        hbox_path->Show(false);
        hbox_path->Layout();

        hbox_width->Show(true);
        hbox_width->Layout();

        hbox_height->Show(true);
        hbox_height->Layout();

        hbox_channels->Show(false);
        hbox_channels->Layout();
    }
    vbox->Layout();
}

//TODO: complete
void VideoSourceFrame::OnVideoSourceFrameButtonOk(wxCommandEvent& event) {
    this->Hide();

    int source_type = ch_source_type->GetSelection();

    int source_id = ch_devices->GetSelection();

    if (source_type == 2) {
        tc->SetValue(wxT("dummy"));
    } else if (source_type == 3) {
        tc->SetValue(wxT("desktop"));
    }

    wxString str = tc->GetValue();
    tc->SetValue(wxT(""));

    struct video_source* vs;
    if (node_id == -1) {
        vs = new video_source();
        vs->smb = nullptr;
        vs->gmb = nullptr;
        vs->mats = nullptr;
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        vs = (struct video_source*)agn->component;

        video_source_close(vs);
    }

    if (source_type == 0) {
        video_source_init(vs, source_id);
    } else {
        video_source_init(vs, str.c_str().AsChar());
    }

    wxString str_width = tc_width->GetValue();
    tc_width->SetValue(wxT(""));
    if (strlen(str_width.c_str().AsChar()) > 0) {
        vs->video_width = stoi(str_width.c_str().AsChar());
    }

    wxString str_height = tc_height->GetValue();
    tc_height->SetValue(wxT(""));
    if (strlen(str_height.c_str().AsChar()) > 0) {
        vs->video_height = stoi(str_height.c_str().AsChar());
    }

    wxString str_channels = tc_channels->GetValue();
    tc_channels->SetValue(wxT(""));
    if (strlen(str_channels.c_str().AsChar()) > 0) {
        vs->video_channels = stoi(str_channels.c_str().AsChar());
    }
    vs->smb_size_req = vs->video_width * vs->video_height * vs->video_channels;

    wxString str_direction = ch_direction->GetStringSelection();
    ch_direction->SetSelection(0);
    if (str_direction.starts_with("no")) {
        vs->do_copy = false;
        vs->direction_smb_to_gmb = true;
    } else if (str_direction.starts_with("Host->GPU")) {
        vs->do_copy = true;
        vs->direction_smb_to_gmb = true;
    } else if (str_direction.starts_with("GPU->Host")){
        vs->do_copy = true;
        vs->direction_smb_to_gmb = false;
    }
    
    if (node_id == -1) {
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        video_source_ui_graph_init(agn, (application_graph_component)vs, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    }
    myApp->drawPane->Refresh();
}

void VideoSourceFrame::OnVideoSourceFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    ch_source_type->SetSelection(0);
    tc->SetValue(wxT(""));
    tc_width->SetValue(wxT(""));
    tc_height->SetValue(wxT(""));
    tc_channels->SetValue(wxT(""));
    ch_direction->SetSelection(0);
}

void VideoSourceFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct video_source* vs = (struct video_source*)agn->component;
        ch_source_type->SetSelection(vs->source_type);

        tc->SetValue(wxString(vs->name));

        stringstream s_w;
        s_w << vs->video_width;
        tc_width->SetValue(wxString(s_w.str()));

        stringstream s_h;
        s_h << vs->video_height;
        tc_height->SetValue(wxString(s_h.str()));

        stringstream s_c;
        s_c << vs->video_channels;
        tc_channels->SetValue(wxString(s_c.str()));

        if (!vs->do_copy && vs->direction_smb_to_gmb) {
            ch_direction->SetSelection(0);
        } else if (vs->do_copy && vs->direction_smb_to_gmb) {
            ch_direction->SetSelection(1);
        } else if (vs->do_copy && !vs->direction_smb_to_gmb) {
            ch_direction->SetSelection(2);
        }
        wxCommandEvent dummy;
        VideoSourceFrame::OnSourceTypeChange(dummy);
    }
    wxFrame::Show(true);
}