#include "MaskRCNNUI.h"

#include "MaskRCNN.h"

#include "MainUI.h"
#include <fstream>
#include <sstream>

const string TEXT_VIDEO_SOURCE_INPUT = "Video Source Input";
const string TEXT_VIDEO_SOURCE_OUTPUT = "Video Source Output";

void mask_rcnn_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_MASK_RCNN;

    agn->name = "Mask RCNN";
    
    pair<enum application_graph_component_type, void*> inner_out = pair<enum application_graph_component_type, void*>(AGCT_MASK_RCNN, (void*)agc);
    agn->outputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(-1, inner_out));

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct mask_rcnn* mrcnn = (struct mask_rcnn*)agn->component;

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mrcnn->net_conf_threshold));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_FLOAT, (void*)&mrcnn->net_mask_threshold));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING_LIST, (void*)&mrcnn->net_classes_active));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_INPUT));
    pair<enum application_graph_component_type, void*> inner_in_1 = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&mrcnn->v_src_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in_1));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_OUTPUT));
    pair<enum application_graph_component_type, void*> inner_in_2 = pair<enum application_graph_component_type, void*>(AGCT_SHARED_MEMORY_BUFFER, (void*)&mrcnn->v_src_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in_2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = mask_rcnn_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = nullptr;
    agn->on_input_edit = nullptr;
    agn->on_delete = mask_rcnn_destroy;
    agn->externalise = mask_rcnn_externalise;
}

MaskRCNNFrame::MaskRCNNFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Mask RCNN")) {

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);
    

    wxBoxSizer* hbox_confthres = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_confthres = new wxStaticText(panel, -1, wxT("Confidence Threshold"));
    hbox_confthres->Add(st_confthres, 0, wxRIGHT, 8);

    tc_confthres = new wxTextCtrl(panel, -1, wxT("0.5"));
    hbox_confthres->Add(tc_confthres, 1);

    vbox->Add(hbox_confthres, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_maskthres = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_maskthres = new wxStaticText(panel, -1, wxT("Size"));
    hbox_maskthres->Add(st_maskthres, 0, wxRIGHT, 8);

    tc_maskthres = new wxTextCtrl(panel, -1, wxT("0.3"));
    hbox_maskthres->Add(tc_maskthres, 1);

    vbox->Add(hbox_maskthres, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    wxBoxSizer* hbox_classes = new wxBoxSizer(wxHORIZONTAL);
    wxStaticText* st_classes = new wxStaticText(panel, wxID_ANY,
        wxT("Active Classes"));

    hbox_classes->Add(st_classes, 0);
    vbox->Add(hbox_classes, 0, wxLEFT | wxTOP, 10);

    vbox->Add(-1, 10);

    
        stringstream ss_classes;

        string classes_file = "./data/mask_rcnn/mscoco_labels.names";
        ifstream ifs(classes_file.c_str());
        string line;
        int line_c = 0;
        while (getline(ifs, line)) {
            if (strlen(line.c_str()) > 0) {
                if (line_c > 0) ss_classes << ",";
                ss_classes << line;
                line_c++;
            }
        }
        classes = ss_classes.str();

    wxBoxSizer* hbox_classes_t = new wxBoxSizer(wxHORIZONTAL);
    tc_classes = new wxTextCtrl(panel, wxID_ANY, wxString(classes),
        wxPoint(-1, -1), wxSize(-1, -1), wxTE_MULTILINE);
    tc_classes->SetMinSize(wxSize(200, 200));

    hbox_classes_t->Add(tc_classes, 1, wxEXPAND);
    vbox->Add(hbox_classes_t, 1, wxLEFT | wxRIGHT | wxEXPAND, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    wxButton* ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &MaskRCNNFrame::OnMaskRCNNFrameButtonOk, this);
    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &MaskRCNNFrame::OnMaskRCNNFrameButtonClose, this);

    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void MaskRCNNFrame::OnMaskRCNNFrameButtonOk(wxCommandEvent& event) {
    this->Hide();

    struct mask_rcnn* mrcnn;
    if (node_id == -1) {
        mrcnn = new mask_rcnn();
        mask_rcnn_init(mrcnn);
    } else {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        mrcnn = (struct mask_rcnn*)agn->component;
        mrcnn->net_classes_active.clear();
    }
    wxString confthres = tc_confthres->GetValue();
    tc_confthres->SetValue("0.5");
    mrcnn->net_conf_threshold = stof(confthres.c_str().AsChar());
    wxString maskthres = tc_maskthres->GetValue();
    tc_maskthres->SetValue("0.3");
    mrcnn->net_mask_threshold = stof(maskthres.c_str().AsChar());

    wxString classes_l = tc_classes->GetValue();
    tc_classes->SetValue(classes);
    int start = 0;
    int end = classes_l.find_first_of(",", start);
    while (end != wxString::npos) {
        mrcnn->net_classes_active.push_back(string((classes_l.substr(start, end-start)).c_str().AsChar()));
        start = end+1;
        end = classes_l.find_first_of(",", start);
    }
    mrcnn->net_classes_active.push_back(string((classes_l.substr(start, end)).c_str().AsChar()));

    if (node_id == -1) {
        application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        mask_rcnn_ui_graph_init(agn, (application_graph_component)mrcnn, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    }
    myApp->drawPane->Refresh();
}

void MaskRCNNFrame::OnMaskRCNNFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc_confthres->SetValue("0.5");
    tc_maskthres->SetValue("0.3");
    tc_classes->SetValue(classes);
}

void MaskRCNNFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct mask_rcnn* mrcnn = (struct mask_rcnn*)agn->component;

        stringstream s_ct;
        s_ct << mrcnn->net_conf_threshold;
        tc_confthres->SetValue(wxString(s_ct.str()));

        stringstream s_mt;
        s_mt << mrcnn->net_mask_threshold;
        tc_maskthres->SetValue(wxString(s_mt.str()));

        stringstream s_classes;
        for (int i = 0; i < mrcnn->net_classes_active.size(); i++) {
            if (i > 0) {
                s_classes << ",";
            }
            s_classes << mrcnn->net_classes_active[i];
        }
        
        tc_classes->SetValue(wxString(s_classes.str().c_str()));
    }
    wxFrame::Show(true);
}