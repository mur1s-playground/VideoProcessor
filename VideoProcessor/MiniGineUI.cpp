#include "MiniGineUI.h"
#include "MiniGine.h"

#include "MainUI.h"
#include "Logger.h"

const string TEXT_VIDEO_SOURCE_IN = "Video Source IN";
const string TEXT_VIDEO_SOURCE_OUT = "Video Source OUT";

void mini_gine_ui_graph_init(struct application_graph_node* agn, application_graph_component agc, int pos_x, int pos_y) {
    agn->component = agc;
    agn->component_type = AGCT_MINI_GINE;

    agn->name = "Mini Gine";

    agn->pos_x = pos_x;
    agn->pos_y = pos_y;

    struct mini_gine* mg = (struct mini_gine*)agc;
    
    //agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    //agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&mg->config_path));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_IN));

    pair<enum application_graph_component_type, void*> inner_in = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&mg->v_src_in);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));
    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_STRING, (void*)&TEXT_VIDEO_SOURCE_OUT));

    pair<enum application_graph_component_type, void*> inner_in2 = pair<enum application_graph_component_type, void*>(AGCT_VIDEO_SOURCE, (void*)&mg->v_src_out);
    agn->inputs.push_back(pair<int, pair<enum application_graph_component_type, void*>>(agn->v.size() - 1, inner_in2));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_SEPARATOR, nullptr));

    agn->v.push_back(pair<enum application_graph_node_vtype, void*>(AGNVT_INT, (void*)&agn->process_tps_balancer.sleep_ms));

    application_graph_tps_balancer_init(agn, 30);

    agn->process = mini_gine_loop;
    agn->process_run = false;
    agn->on_input_connect = nullptr;
    agn->on_input_disconnect = mini_gine_on_input_disconnect;
    agn->on_input_edit = nullptr;
    agn->externalise = mini_gine_externalise;
    agn->on_delete = mini_gine_destroy;
}

MiniGineFrame::MiniGineFrame(wxWindow* parent) : wxFrame(parent, -1, wxT("Mini Gine")) {
    node_graph_id = -1;
    node_id = -1;

    wxPanel* panel = new wxPanel(this, -1);

    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    wxBoxSizer* hbox_name = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_name = new wxStaticText(panel, -1, wxT("Config Path"));
    hbox_name->Add(st_name, 0, wxRIGHT, 8);

    tc = new wxTextCtrl(panel, -1, wxT(""));
    hbox_name->Add(tc, 1);

    vbox->Add(hbox_name, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);

    vbox->Add(-1, 10);

    wxBoxSizer* hbox_buttons = new wxBoxSizer(wxHORIZONTAL);

    ok_button = new wxButton(panel, -1, wxT("Ok"), wxDefaultPosition, wxSize(70, 30));
    ok_button->Bind(wxEVT_BUTTON, &MiniGineFrame::OnMiniGineFrameButtonOk, this);

    hbox_buttons->Add(ok_button, 0);

    wxButton* close_button = new wxButton(panel, -1, wxT("Close"), wxDefaultPosition, wxSize(70, 30));
    close_button->Bind(wxEVT_BUTTON, &MiniGineFrame::OnMiniGineFrameButtonClose, this);
    hbox_buttons->Add(close_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_buttons, 0, wxALIGN_RIGHT | wxRIGHT, 10);

    /* ENTITY SETTINGS */
    wxBoxSizer* hbox_esettings = new wxBoxSizer(wxVERTICAL);

    wxStaticText* st_es = new wxStaticText(panel, -1, wxT("Entity Settings"));
    hbox_esettings->Add(st_es, 0, wxRIGHT, 8);

    tc_entity_settings = new wxTextCtrl(panel, wxID_ANY, wxT(""), wxPoint(-1, -1), wxSize(-1, -1), wxTE_MULTILINE);
    tc_entity_settings->SetMinSize(wxSize(200, 200));
    hbox_esettings->Add(tc_entity_settings, 1, wxEXPAND);

    wxButton* apply_button = new wxButton(panel, -1, wxT("Apply Settings"), wxDefaultPosition, wxSize(70, 30));
    apply_button->Bind(wxEVT_BUTTON, &MiniGineFrame::OnMiniGineFrameButtonApplyEntitySettings, this);
    hbox_esettings->Add(apply_button, 0, wxLEFT | wxBOTTOM, 10);

    vbox->Add(hbox_esettings, 1, wxLEFT | wxRIGHT | wxEXPAND, 10);

    /* ENTITY GROUP SETTINGS */
    wxBoxSizer* hbox_egsettings = new wxBoxSizer(wxHORIZONTAL);

    wxStaticText* st_esg = new wxStaticText(panel, -1, wxT("Entity Group Settings"));
    hbox_egsettings->Add(st_esg, 0, wxRIGHT, 8);

    tc_entity_group_settings = new wxTextCtrl(panel, -1, wxT("0,1,1,0,0,255,255,255,1,1,1,"));
    hbox_egsettings->Add(tc_entity_group_settings, 1);

    wxButton* apply_button2 = new wxButton(panel, -1, wxT("Apply Group Settings"), wxDefaultPosition, wxSize(70, 30));
    apply_button2->Bind(wxEVT_BUTTON, &MiniGineFrame::OnMiniGineFrameButtonApplyEntityGroupSettings, this);
    hbox_egsettings->Add(apply_button2, 1);

    vbox->Add(hbox_egsettings, 1, wxLEFT | wxRIGHT, 10);

    panel->SetSizer(vbox);
}

void MiniGineFrame::OnMiniGineFrameButtonApplyEntityGroupSettings(wxCommandEvent& event) {
    wxString es = tc_entity_group_settings->GetValue();

    struct mini_gine* mg = (struct mini_gine*)ags[node_graph_id]->nodes[node_id]->component;

    int start = 0;
    int end = es.find_first_of(",", start);
    int ct = 0;

    int e_start = 0;
    int e_end = 0;
    float s_mult = 1.0f;
    float s_fall = 0.0f;
    int anim_type = 0;
    struct mini_gine_model_params mgmp_c = { 255, 255, 255, 1.0f };
    vector<float> params;

    vector3<float> anim_type_params_2 = { 1.0f, 1.0f, 1.0f };

    while (end != wxString::npos) {
        string tmp((es.substr(start, end - start)).c_str().AsChar());

        switch (ct) {
        case 0:
            if (tmp.compare("r") == 0) {
                e_start = (rand() / (float)RAND_MAX) * (mg->entities.size() - 1);
            } else {
                e_start = stoi(tmp);
            }
            ct++;
            break;
        case 1:
            if (tmp.compare("r") == 0) {
                e_end = e_start + (int)roundf((rand() / (float)RAND_MAX) * (mg->entities.size() - e_start));
            } else {
                e_end = stoi(tmp);
            }
            ct++;
            break;
        case 2:
            if (tmp.compare("r") == 0) {
                s_mult = (rand() / (float)RAND_MAX) * 15.0f;
            } else {
                s_mult = stof(tmp);
            }
            ct++;
            break;
        case 3:
            if (tmp.compare("r") == 0) {
                s_fall = (rand() / (float)RAND_MAX) * 10.0f;
            } else {
                s_fall = stof(tmp);
            }
            ct++;
            break;
        case 4:
            if (tmp.compare("r") == 0) {
                anim_type = (int)roundf((rand() / (float)RAND_MAX) * 4.0f);
            } else {
                anim_type = stoi(tmp);
            }
            ct++;
            break;
        case 5:
            if (tmp.compare("r") == 0) {
                mgmp_c.r = (int)roundf((rand() / (float)RAND_MAX) * 255);
            } else {
                mgmp_c.r = stoi(tmp);
            }
            ct++;
            break;
        case 6:
            if (tmp.compare("r") == 0) {
                mgmp_c.g = (int)roundf((rand() / (float)RAND_MAX) * 255);
            }
            else {
                mgmp_c.g = stoi(tmp);
            }
            ct++;
            break;
        case 7:
            if (tmp.compare("r") == 0) {
                mgmp_c.b = (int)roundf((rand() / (float)RAND_MAX) * 255);
            }
            else {
                mgmp_c.b = stoi(tmp);
            }
            ct++;
            break;
        case 8:
            if (tmp.compare("r") == 0) {
                mgmp_c.s = (int)roundf((rand() / (float)RAND_MAX) * 255);
            }
            else {
                mgmp_c.s = stoi(tmp);
            }
            ct++;
            break;
        default:
            if (tmp.compare("r") == 0) {
                params.push_back((int)roundf((rand() / (float)RAND_MAX) * 255));
            } else {
                params.push_back(stof(tmp));
            }
            ct++;
            break;
        }
        start = end + 1;
        end = es.find_first_of(",", start);
    }

    for (int e = e_start; e <= e_end; e++) {
        mg->entities[e].model_params_s_multiplier = s_mult;
        mg->entities[e].model_params_s_falloff = s_fall;

        mg->entities_meta[e].mgmp.clear();
        mg->entities_meta[e].mgema.clear();
        if (anim_type == MGEAT_MOVIE) {
            for (int p = 0; p < mg->models[mg->entities[e].model_id].model_params; p++) {
                struct mini_gine_entity_meta_animation mgema;
                mgema.animation_type = MGEAT_MOVIE;
                mg->entities_meta[e].mgema.push_back(mgema);
            }
        } else if (anim_type == MGEAT_JUST_ON) {
            for (int p = 0; p < mg->models[mg->entities[e].model_id].model_params; p++) {
                mg->entities_meta[e].mgmp.push_back(mgmp_c);

                struct mini_gine_entity_meta_animation mgema;
                mgema.animation_type = MGEAT_JUST_ON;
                mg->entities_meta[e].mgema.push_back(mgema);
            }
        } else if (anim_type == MGEAT_2COLOR_BLINK) {
            for (int p = 0; p < mg->models[mg->entities[e].model_id].model_params; p++) {
                mg->entities_meta[e].mgmp.push_back(mgmp_c);

                struct mini_gine_entity_meta_animation mgema;
                mgema.animation_type = MGEAT_2COLOR_BLINK;
                mgema.animation_params = params;
                mg->entities_meta[e].mgema.push_back(mgema);
            }
        } else if (anim_type == MGEAT_SNAKE) {
            for (int p = 0; p < mg->models[mg->entities[e].model_id].model_params; p++) {
                mg->entities_meta[e].mgmp.push_back(mgmp_c);

                struct mini_gine_entity_meta_animation mgema;
                mgema.animation_type = MGEAT_SNAKE;
                mgema.animation_params = params;
                mg->entities_meta[e].mgema.push_back(mgema);
            }
        }
    }
    stringstream ss_s;

    for (int e = 0; e < mg->entities.size(); e++) {
        int model_id = mg->entities[e].model_id;
        int model_params = mg->models[model_id].model_params;
        ss_s << e << "," << mg->entities[e].model_params_s_multiplier << "," << mg->entities[e].model_params_s_falloff << "," << model_id << "," << model_params << ",";
        for (int p = 0; p < model_params; p++) {
            ss_s << (int)mg->entities_meta[e].mgmp[p].r << "," << (int)mg->entities_meta[e].mgmp[p].g << "," << (int)mg->entities_meta[e].mgmp[p].b << "," << mg->entities_meta[e].mgmp[p].s << ",";
            ss_s << mg->entities_meta[e].mgema[p].animation_type << "," << mg->entities_meta[e].mgema[p].animation_params.size() << ",";
            for (int ap = 0; ap < mg->entities_meta[e].mgema[p].animation_params.size(); ap++) {
                ss_s << mg->entities_meta[e].mgema[p].animation_params[ap] << ",";
            }
        }
        ss_s << std::endl;
    }
    wxString ess(ss_s.str());
    tc_entity_settings->SetValue(ess);

    mini_gine_on_entity_update(mg);
}

void MiniGineFrame::OnMiniGineFrameButtonApplyEntitySettings(wxCommandEvent& event) {
    wxString es = tc_entity_settings->GetValue();
    
    struct mini_gine* mg = (struct mini_gine *)ags[node_graph_id]->nodes[node_id]->component;
    
    int e = 0;
    int ct = 0;

    stringstream ss;
    ss << es.c_str().AsChar();

    string line;
    while (std::getline(ss, line)) {
        if (strlen(line.c_str()) == 0) break;
        int start = 0;
        int end = line.find_first_of(",", start);
        int ct = 0;
        int model_params = 0;
        int mp = 0;
        int ap_size = 0;
        int ap = 0;
        int p = 0;
        //logger(line);
        while (end != string::npos) {
            string tmp(line.substr(start, end - start));
            if (strlen(tmp.c_str()) == 0) break;
            switch (ct) {
                case 0:
                    e = stoi(tmp);
                    ct++;
                    break;
                case 1:
                    mg->entities[e].model_params_s_multiplier = stof(tmp);
                    ct++;
                    break;
                case 2:
                    mg->entities[e].model_params_s_falloff = stof(tmp);
                    ct++;
                    break;
                case 3:
                    mg->entities[e].model_id = stoi(tmp);
                    ct++;
                    break;
                case 4:
                    model_params = stoi(tmp);
                    ct++;
                    break;
                default:
                    if (model_params > 0) {
                        switch (mp) {
                        case 0:
                            mg->entities_meta[e].mgmp[p].r = stoi(tmp);
                            mp++;
                            break;
                        case 1:
                            mg->entities_meta[e].mgmp[p].g = stoi(tmp);
                            mp++;
                            break;
                        case 2:
                            mg->entities_meta[e].mgmp[p].b = stoi(tmp);
                            mp++;
                            break;
                        case 3:
                            mg->entities_meta[e].mgmp[p].s = stof(tmp);
                            mp++;
                            break;
                        case 4:
                            mg->entities_meta[e].mgema[p].animation_type = (enum mini_gine_entity_animation_types)stoi(tmp);
                            mp++;
                            break;
                        case 5:
                            ap_size = stoi(tmp);
                            if (ap_size == 0) {
                                p++;
                                mp = 0;
                            } else {
                                mp++;
                            }
                            break;
                        default:
                            if (ap < ap_size) {
                                if (mg->entities_meta[e].mgema[p].animation_params.size() <= ap) {
                                    mg->entities_meta[e].mgema[p].animation_params.push_back(stof(tmp));
                                } else {
                                    mg->entities_meta[e].mgema[p].animation_params[ap] = stof(tmp);
                                }
                                ap++;
                                if (ap == ap_size) {
                                    ap = 0;
                                    mp = 0;
                                    p++;
                                }
                            }
                            break;
                        }
                    }
                    break;
            }
            start = end + 1;
            end = line.find_first_of(",", start);
        }
    }

    mini_gine_on_entity_update(mg);
}

void MiniGineFrame::OnMiniGineFrameButtonOk(wxCommandEvent& event) {
    this->Hide();
    wxString str = tc->GetValue();
    tc->SetValue(wxT(""));
    if (node_id == -1) {
        struct mini_gine* mg = new mini_gine();
        mini_gine_init(mg, str.c_str().AsChar());
        struct application_graph_node* agn = new application_graph_node();
        agn->n_id = ags[node_graph_id]->nodes.size();
        mini_gine_ui_graph_init(agn, (application_graph_component)mg, myApp->drawPane->right_click_mouse_x, myApp->drawPane->right_click_mouse_y);
        ags[node_graph_id]->nodes.push_back(agn);
    } else {
        //TODO:
    }
    myApp->drawPane->Refresh();
}

void MiniGineFrame::OnMiniGineFrameButtonClose(wxCommandEvent& event) {
    this->Hide();
    tc->SetValue(wxT(""));
    tc_entity_settings->SetValue(wxT(""));
}

void MiniGineFrame::Show(int node_graph_id, int node_id) {
    this->node_graph_id = node_graph_id;
    this->node_id = node_id;
    if (node_id > -1) {
        struct application_graph_node* agn = ags[node_graph_id]->nodes[node_id];
        struct mini_gine* mg = (struct mini_gine*)agn->component;
        tc->SetValue(wxString(mg->config_path));
        stringstream ss_s;
        
        for (int e = 0; e < mg->entities.size(); e++) {
            int model_id = mg->entities[e].model_id;
            int model_params = mg->models[model_id].model_params;
            ss_s << e << "," << mg->entities[e].model_params_s_multiplier << "," << mg->entities[e].model_params_s_falloff << "," << model_id << "," << model_params << ",";
            for (int p = 0; p < model_params; p++) {
                ss_s << (int)mg->entities_meta[e].mgmp[p].r << "," << (int)mg->entities_meta[e].mgmp[p].g << "," << (int)mg->entities_meta[e].mgmp[p].b << "," << mg->entities_meta[e].mgmp[p].s << ",";
                ss_s << mg->entities_meta[e].mgema[p].animation_type << "," << mg->entities_meta[e].mgema[p].animation_params.size() << "," ;
                for (int ap = 0; ap < mg->entities_meta[e].mgema[p].animation_params.size(); ap++) {
                    ss_s << mg->entities_meta[e].mgema[p].animation_params[ap] << ",";
                }
            }
            ss_s << std::endl;
        }
        wxString es(ss_s.str());
        tc_entity_settings->SetValue(es);
        
    }
    wxFrame::Show(true);
}