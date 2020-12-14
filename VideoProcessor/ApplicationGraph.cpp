#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

#include "VideoSource.h"
#include "VideoSourceUI.h"

#include "SharedMemoryBuffer.h"
#include "SharedMemoryBufferUI.h"

#include "MaskRCNN.h"
#include "MaskRCNNUI.h"

#include "ImShow.h"
#include "ImShowUI.h"

#include "GPUComposerElement.h"
#include "GPUComposerElementUI.h"

#include "GPUComposer.h"
#include "GPUComposerUI.h"

#include "GPUDenoise.h"
#include "GPUDenoiseUI.h"

#include "GPUVideoAlphaMerge.h"
#include "GPUVideoAlphaMergeUI.h"

#include "GPUMotionBlur.h"
#include "GPUMotionBlurUI.h"

#include "GPUGaussianBlur.h"
#include "GPUGaussianBlurUI.h"

#include "GPUEdgeFilter.h"
#include "GPUEdgeFilterUI.h"

#include "GPUPaletteFilter.h"
#include "GPUPaletteFilterUI.h"

#include "GPUAudioVisual.h"
#include "GPUAudioVisualUI.h"

#include "AudioSource.h"
#include "AudioSourceUI.h"

#include "MiniGine.h"
#include "MiniGineUI.h"

#include "MainUI.h"

#include "Logger.h"

using namespace std;

vector <struct application_graph *>   ags;
int application_graph_active_id = 0;
int application_graph_hovering_node_id = -1;

unsigned long long application_graph_tps_balancer_get_time() {
    struct timespec now;
    clock_gettime(0, &now);
    return now.tv_sec * 1000000000 + now.tv_nsec;
}

void application_graph_tps_balancer_init(struct application_graph_node *agn, int tps_target) {
    agn->process_tps_balancer.tps_current = tps_target;
    agn->process_tps_balancer.tps_target = tps_target;
    agn->process_tps_balancer.overlap = 0;
    agn->process_tps_balancer.started = false;
}

void application_graph_tps_balancer_timer_start(struct application_graph_node* agn) {
    if (!agn->process_tps_balancer.started) {
        agn->process_tps_balancer.started = true;
        clock_gettime(0, &agn->process_tps_balancer.start);
    }
}

void application_graph_tps_balancer_timer_stop(struct application_graph_node* agn) {
    agn->process_tps_balancer.started = false;
    clock_gettime(0, &agn->process_tps_balancer.stop);
}

int application_graph_tps_balancer_get_sleep_ms(struct application_graph_node* agn) {
    unsigned long long time_start = agn->process_tps_balancer.start.tv_sec * 1000000000 + agn->process_tps_balancer.start.tv_nsec;
    unsigned long long time_stop = agn->process_tps_balancer.stop.tv_sec * 1000000000 + agn->process_tps_balancer.stop.tv_nsec;
    unsigned long long time_delta = time_stop - time_start;
    int sleep_time = (int)floor(((1000000000 / agn->process_tps_balancer.tps_target) - time_delta) / 100000);
    if (sleep_time > 0) {
        agn->process_tps_balancer.overlap += (sleep_time % 10);
        sleep_time /= 10;
        if (agn->process_tps_balancer.overlap >= 10) {
            sleep_time++;
            agn->process_tps_balancer.overlap -= 10;
        }
        return sleep_time;
    }
    return 0;
}

void application_graph_tps_balancer_sleep(struct application_graph_node* agn) {
    int sleep_time = application_graph_tps_balancer_get_sleep_ms(agn);
    agn->process_tps_balancer.sleep_ms = sleep_time;
    if (sleep_time > 0) {
        Sleep(sleep_time);
    }
}

int application_graph_is_on_input(int id, int node_id, int pos_x, int pos_y, float* dist_out) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    float closest_dist = -1.0f;
    int closest_id = -1;
    if (current_node != nullptr) {
        for (int in = 0; in < current_node->inputs.size(); in++) {
            int to_x = current_node->pos_x;
            int to_height_id = current_node->inputs[in].first + 1;
            int h_1 = current_node->heights[to_height_id];
            int h_2 = current_node->heights[to_height_id + 1];
            int to_y = h_1 + (h_2 - h_1) / 2;

            float dist_2 = sqrt(pow(to_x - pos_x, 2) + pow(to_y - pos_y, 2));
            if ((dist_2 < closest_dist || closest_dist < 0) && dist_2 < 6) {
                closest_dist = dist_2;
                closest_id = in;
            }
        }
    }
    if (dist_out != nullptr) {
        *dist_out = closest_dist;
    }
    return closest_id;
}

int application_graph_is_on_output(int id, int node_id, int pos_x, int pos_y, float* dist_out) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    float closest_dist = -1.0f;
    int closest_id = -1;
    if (current_node != nullptr) {
        for (int out = 0; out < current_node->outputs.size(); out++) {
            int from_x = current_node->pos_x + 5 + current_node->max_width + 5;
            int from_height_id = current_node->outputs[out].first + 1;
            int h_1 = current_node->heights[from_height_id];
            int h_2 = current_node->heights[from_height_id + 1];
            int from_y = h_1 + (h_2 - h_1) / 2;

            float dist_1 = sqrt(pow(from_x - pos_x, 2) + pow(from_y - pos_y, 2));
            if ((dist_1 < closest_dist || closest_dist < 0) && dist_1 < 6) {
                closest_dist = dist_1;
                closest_id = out;
            }
        }
    }
    if (dist_out != nullptr) {
        *dist_out = closest_dist;
    }
    return closest_id;
}

void application_graph_add_edge_intl(int id, int closest_nid_1, int closest_nid_2, int closest_id_1, int closest_id_2) {
    struct application_graph_edge* edge = new application_graph_edge();
    pair<application_graph_node*, int> from;
    pair<application_graph_node*, int> to;
    edge->from = pair<application_graph_node*, int>(ags[id]->nodes[closest_nid_1], closest_id_1);
    edge->to = pair<application_graph_node*, int>(ags[id]->nodes[closest_nid_2], closest_id_2);
    ags[id]->edges.push_back(edge);

    struct application_graph_node* current_node = ags[id]->nodes[closest_nid_2];
    struct application_graph_edge* current_edge = edge;

    void* input_target_ptr = current_node->inputs[closest_id_2].second.second;

    if (current_node->inputs[closest_id_2].second.first == AGCT_SHARED_MEMORY_BUFFER) {
        struct shared_memory_buffer** input_target_ptr_t = (struct shared_memory_buffer**)input_target_ptr;
        *input_target_ptr_t = (struct shared_memory_buffer*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }
    else if (current_node->inputs[closest_id_2].second.first == AGCT_GPU_MEMORY_BUFFER) {
        struct gpu_memory_buffer** input_target_ptr_t = (struct gpu_memory_buffer**)input_target_ptr;
        *input_target_ptr_t = (struct gpu_memory_buffer*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }
    else if (current_node->inputs[closest_id_2].second.first == AGCT_VIDEO_SOURCE) {
        struct video_source** input_target_ptr_t = (struct video_source**)input_target_ptr;
        *input_target_ptr_t = (struct video_source*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }
    else if (current_node->inputs[closest_id_2].second.first == AGCT_MASK_RCNN) {
        struct mask_rcnn** input_target_ptr_t = (struct mask_rcnn**)input_target_ptr;
        *input_target_ptr_t = (struct mask_rcnn*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }
    else if (current_node->inputs[closest_id_2].second.first == AGCT_GPU_COMPOSER_ELEMENT) {
        struct gpu_composer_element** input_target_ptr_t = (struct gpu_composer_element**)input_target_ptr;
        *input_target_ptr_t = (struct gpu_composer_element*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }
    else if (current_node->inputs[closest_id_2].second.first == AGCT_AUDIO_SOURCE) {
        struct audio_source** input_target_ptr_t = (struct audio_source**)input_target_ptr;
        *input_target_ptr_t = (struct audio_source*)(current_edge->from.first->outputs[current_edge->from.second].second.second);
    }

    if (current_node->on_input_connect != nullptr) {
#ifndef FPTR_OIC_AGN_INT
#define FPTR_OIC_AGN_INT
        typedef void (*oicptr)(struct application_graph_node* agn, int);
#endif
        oicptr f_ptr = (oicptr)current_node->on_input_connect;
        f_ptr(current_node, closest_id_2);
    }
}

void application_graph_add_edge(int id, int pos_x1, int pos_y1, int pos_x2, int pos_y2) {
    int closest_nid_1 = -1;
    int closest_id_1 = -1;
    float closest_dist_1 = -1.0f;

    int closest_nid_2 = -1;
    int closest_id_2 = -1;
    float closest_dist_2 = -1.0f;
    for (int i = 0; i < ags[id]->nodes.size(); i++) {
        struct application_graph_node* current_node = ags[id]->nodes[i];

        if (current_node == nullptr) continue;

        float closest_dist_out = -1.0f;
        int closest_id_out = application_graph_is_on_output(id, i, pos_x1, pos_y1, &closest_dist_out);
        if (closest_id_out > -1 && (closest_dist_out < closest_dist_1 || closest_dist_1 < 0) && closest_dist_out < 6) {
            closest_dist_1 = closest_dist_out;
            closest_id_1 = closest_id_out;
            closest_nid_1 = i;
        }

        float closest_dist_in = -1.0f;
        int closest_id_in = application_graph_is_on_input(id, i, pos_x2, pos_y2, &closest_dist_in);
        if (closest_id_in > -1 && (closest_dist_in < closest_dist_2 || closest_dist_2 < 0) && closest_dist_in < 6) {
            closest_dist_2 = closest_dist_in;
            closest_id_2 = closest_id_in;
            closest_nid_2 = i;
        }

    }

    if (closest_nid_1 >= 0 && closest_nid_2 >= 0 && closest_id_1 >= 0 && closest_id_2 >= 0) {
        application_graph_add_edge_intl(id, closest_nid_1, closest_nid_2, closest_id_1, closest_id_2);
    }
}

void application_graph_start_stop_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    if (current_node != nullptr) {
        if (current_node->process_run) {
            application_graph_stop_node(id, node_id);
        } else {
            application_graph_start_node(id, node_id);
        }
    }
}

void application_graph_start_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];

    if (current_node->process != nullptr) {
        current_node->process_run = true;
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)current_node->process, current_node, 0, 0);
        myApp->drawPane->Refresh();
    }
}

void application_graph_stop_node(int id, int node_id) {
    struct application_graph_node* current_node = ags[id]->nodes[node_id];
    current_node->process_run = false;
}

void application_graph_draw_nodes(struct application_graph* ag, wxDC& dc) {
    wxFont f_normal = dc.GetFont();
    wxFont f_bold = dc.GetFont();
    f_bold.MakeBold();

    for (int i = 0; i < ag->nodes.size(); i++) {
        struct application_graph_node *current_node = ag->nodes[i];

        if (current_node == nullptr) continue;
        dc.SetFont(f_bold);
        wxSize max_width = dc.GetTextExtent(current_node->name);
        max_width.SetWidth(max_width.GetWidth() + 5);
        int total_height = max_width.GetHeight();

        dc.SetFont(f_normal);
        for (int j = 0; j < current_node->v.size(); j++) {
            pair<enum application_graph_node_vtype, void*> current_v = current_node->v[j];
            stringstream ss;
            bool draw_text = true;
            if (current_v.first == AGNVT_STRING) {
                ss << *((string*)current_v.second);
            } else if (current_v.first == AGNVT_STRING_LIST) {
                draw_text = false;
                vector<string> current_string_list = *((vector<string> *) current_v.second);
                stringstream line;
                int line_c = 0;
                for (int sl = 0; sl < current_string_list.size(); sl++) {
                    if (line_c > 0) line << ",";
                    line << current_string_list[sl];
                    line_c++;
                    wxSize tmp = dc.GetTextExtent(line.str());
                    tmp.SetWidth(tmp.GetWidth() + 5);
                    if (tmp.GetWidth() > 200 || sl == current_string_list.size() - 1) {
                        total_height += tmp.GetHeight();
                        if (tmp.GetWidth() > max_width.GetWidth()) {
                            max_width = tmp;
                        }
                        line.str("");
                        line.clear();
                        line_c = 0;
                    }
                }
                
            } else if (current_v.first == AGNVT_INT) {
                int* current_int = (int*)current_v.second;
                ss << *current_int;
            } else if (current_v.first == AGNVT_FLOAT) {
                ss.precision(2);
                float* current_float = (float*)current_v.second;
                ss << *current_float;
            } else if (current_v.first == AGNVT_BOOL) {
                bool* current_bool = (bool*)current_v.second;
                if (*current_bool) {
                    ss << "true";
                } else {
                    ss << "false";
                }
            } else if (current_v.first == AGNVT_SEPARATOR) {
                draw_text = false;
                total_height += 6;
            }
            if (draw_text) {
                wxSize tmp = dc.GetTextExtent(ss.str());
                tmp.SetWidth(tmp.GetWidth() + 5);
                total_height += tmp.GetHeight();
                if (tmp.GetWidth() > max_width.GetWidth()) {
                    max_width = tmp;
                }
            }
        }

        current_node->max_width = max_width.GetWidth();

        int border_pen_size = 2;
        if (application_graph_hovering_node_id == i) {
            border_pen_size--;
        }
        if (current_node->process_run) {
            border_pen_size--;
        }
               
        dc.SetBrush(*wxWHITE_BRUSH);
        dc.SetPen(wxPen(wxColor(0, 0, 0), border_pen_size));
        dc.DrawRectangle(current_node->pos_x, current_node->pos_y, 5 + current_node->max_width + 5, total_height + 5);
        if (application_graph_hovering_node_id == i) {
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.SetPen(wxPen(wxColor(255, 255, 0), 1));
            dc.DrawRectangle(current_node->pos_x, current_node->pos_y, 5 + current_node->max_width + 5, total_height + 5);
        }
        if (current_node->process_run) {
            dc.SetBrush(*wxTRANSPARENT_BRUSH);
            dc.SetPen(wxPen(wxColor(0, 255, 0), 1));
            dc.DrawRectangle(current_node->pos_x+1, current_node->pos_y+1, 5 + current_node->max_width + 3, total_height + 3);
        }
        dc.SetBrush(*wxWHITE_BRUSH);
        dc.SetPen(wxPen(wxColor(0, 0, 0), 1));

        current_node->heights.clear();

        int current_y = current_node->pos_y + 5;
        current_node->heights.push_back(current_y);

        dc.SetFont(f_bold);
        dc.DrawText(wxString(current_node->name), current_node->pos_x + 5, current_y);

        current_y += dc.GetTextExtent(current_node->name).GetHeight();
        current_node->heights.push_back(current_y);

        dc.SetFont(f_normal);
        
        for (int j = 0; j < current_node->v.size(); j++) {
            pair<enum application_graph_node_vtype, void*> current_v = current_node->v[j];
            stringstream ss;
            bool draw_text = true;
            if (current_v.first == AGNVT_STRING) {
                ss << *((string *)current_v.second);
            } else if (current_v.first == AGNVT_STRING_LIST) {
                draw_text = false;
                vector<string> current_string_list = *((vector<string>*) current_v.second);
                stringstream line;
                int line_c = 0;
                for (int sl = 0; sl < current_string_list.size(); sl++) {
                    if (line_c > 0) line << ",";
                    line << current_string_list[sl];
                    line_c++;
                    wxSize tmp = dc.GetTextExtent(line.str());
                    tmp.SetWidth(tmp.GetWidth() + 5);
                    if (tmp.GetWidth() > 200 || sl == current_string_list.size() - 1) {
                        dc.DrawText(wxString(line.str()), current_node->pos_x + 5, current_y);
                        current_y += tmp.GetHeight();
                        line.str("");
                        line.clear();
                        line_c = 0;
                    }
                }
            } else if (current_v.first == AGNVT_INT) {
                int* current_int = (int*)current_v.second;
                ss << *current_int;
            } else if (current_v.first == AGNVT_FLOAT) {
                ss.precision(2);
                float* current_float = (float*)current_v.second;
                ss << *current_float;
            } else if (current_v.first == AGNVT_BOOL) {
                bool* current_bool = (bool*)current_v.second;
                if (*current_bool) {
                    ss << "true";
                }
                else {
                    ss << "false";
                }
            } else if (current_v.first == AGNVT_SEPARATOR) {
                draw_text = false;
                dc.DrawLine(current_node->pos_x+1, current_y, current_node->pos_x + 5 + current_node->max_width + 4, current_y);
                current_y += 2;
            }
            if (draw_text) {
                string tmp = ss.str();
                dc.DrawText(wxString(tmp), current_node->pos_x + 5, current_y);
                current_y += dc.GetTextExtent(tmp).GetHeight();
            }
            current_node->heights.push_back(current_y);
        }

        dc.SetPen(wxPen(wxColor(0, 0, 0), 2));
        for (int j = 0; j < current_node->inputs.size(); j++) {
            pair<int, pair<enum application_graph_component_type, void*>> current_input = current_node->inputs[j];
            int h_1 = current_node->heights[current_input.first + 1];
            int h_2 = current_node->heights[current_input.first + 2];
            dc.DrawCircle(wxPoint(current_node->pos_x, h_1 + (h_2 - h_1)/2), 4);
        }

        for (int j = 0; j < current_node->outputs.size(); j++) {
            pair<int, pair<enum application_graph_component_type, void*>> current_output = current_node->outputs[j];
            int h_1 = current_node->heights[current_output.first + 1];
            int h_2 = current_node->heights[current_output.first + 2];
            dc.DrawCircle(wxPoint(current_node->pos_x + 5 + current_node->max_width + 3, h_1 + (h_2 - h_1) / 2), 4);
        }
    }
}

void application_graph_draw_edges(struct application_graph* ag, wxDC& dc) {
    for (int i = 0; i < ag->edges.size(); i++) {
        struct application_graph_edge* current_edge = ag->edges[i];

        if (current_edge == nullptr) continue;
        int h_1, h_2;

        struct application_graph_node* from = current_edge->from.first;
        int from_id = current_edge->from.second;
        int from_x = from->pos_x + 5 + from->max_width + 5;

        int from_height_id = from->outputs[from_id].first + 1;
        h_1 = from->heights[from_height_id];
        h_2 = from->heights[from_height_id + 1];
        int from_y = h_1 + (h_2 - h_1)/2;

        struct application_graph_node *to = current_edge->to.first;
        int to_id = current_edge->to.second;
        int to_x = to->pos_x;

        int to_height_id = to->inputs[to_id].first + 1;
        h_1 = to->heights[to_height_id];
        h_2 = to->heights[to_height_id + 1];
        int to_y = h_1 + (h_2 - h_1) / 2;

        dc.DrawLine(wxPoint(from_x, from_y), wxPoint(to_x, to_y));
    }
}

void application_graph_hovering_node(int application_graph_id) {
    if (ags.size() > application_graph_id) {
        int hovering_candidate = -1;
        for (int i = 0; i < ags[application_graph_id]->nodes.size(); i++) {
            struct application_graph_node* current_node = ags[application_graph_id]->nodes[i];
            if (current_node == nullptr) continue;
            if (myApp->drawPane->mouse_position_x >= current_node->pos_x && myApp->drawPane->mouse_position_x <= current_node->pos_x + current_node->max_width &&
                myApp->drawPane->mouse_position_y >= current_node->pos_y && current_node->heights.size() > 0 && myApp->drawPane->mouse_position_y <= current_node->heights[current_node->heights.size()-1]) {
                hovering_candidate = i;
            }
        }
        if (hovering_candidate == -1 && application_graph_hovering_node_id > -1) {
            application_graph_hovering_node_id = -1;
            myApp->drawPane->Refresh();
        } else if (application_graph_hovering_node_id != hovering_candidate) {
            application_graph_hovering_node_id = hovering_candidate;
            myApp->drawPane->Refresh();
        }
    }
}

void application_graph_delete_edge(int application_graph_id, int node_id, int input_id) {
    for (int i = 0; i < ags[application_graph_id]->edges.size(); i++) {
        struct application_graph_edge* current_edge = ags[application_graph_id]->edges[i];
        if (current_edge != nullptr) {
            if (current_edge->to.first == ags[application_graph_id]->nodes[node_id] && current_edge->to.second == input_id) {
                application_graph_delete_edge(application_graph_id, i, true);
                break;
            }
        }
    }
}

void application_graph_delete_edge(int application_graph_id, int edge_id, bool refresh) {
    struct application_graph_edge* current_edge = ags[application_graph_id]->edges[edge_id];
    ags[application_graph_id]->edges[edge_id] = nullptr;

#ifndef FPTR_DEL_EDG
#define FPTR_DEL_EDG
    typedef void (*deledgptr)(struct application_graph_edge* edge);
#endif
    if (current_edge->to.first->on_input_disconnect != nullptr) {
        deledgptr f_ptr = (deledgptr)current_edge->to.first->on_input_disconnect;
        f_ptr(current_edge);
    }

    delete current_edge;
    if (refresh) {
        myApp->drawPane->Refresh();
    }
}

void application_graph_delete_node(int application_graph_id, int node_id) {
    struct application_graph_node* current_node = ags[application_graph_id]->nodes[node_id];
    for (int i = 0; i < ags[application_graph_id]->edges.size(); i++) {
        struct application_graph_edge* current_edge = ags[application_graph_id]->edges[i];
        if (current_edge != nullptr) {
            if (current_edge->from.first == current_node || current_edge->to.first == current_node) {
                application_graph_delete_edge(application_graph_id, i, false);
            }
        }
    }
    ags[application_graph_id]->nodes[node_id] = nullptr;
    
#ifndef FPTR_DEL_AGN
#define FPTR_DEL_AGN
    typedef void (*delagnptr)(struct application_graph_node* agn);
#endif
    if (current_node->on_delete != nullptr) {
        delagnptr f_ptr = (delagnptr)current_node->on_delete;
        f_ptr(current_node);
    }

    delete current_node;
    myApp->drawPane->Refresh();
}

void application_graph_node_settings_externalise(struct application_graph_node *agn, string& out_str) {
    stringstream s_out;
    s_out << agn->process_tps_balancer.tps_target << std::endl;
    s_out << agn->start_stop_hotkey << std::endl;

    out_str = s_out.str();
}

void application_graph_node_settings_load(struct application_graph_node *agn, ifstream& in_f) {
    std::string line;
    std::getline(in_f, line);
    agn->process_tps_balancer.tps_target = stoi(line);

    std::getline(in_f, line);
    agn->start_stop_hotkey = stoi(line);
    if (agn->start_stop_hotkey > -1) {
        myApp->drawPane->addHotKey(agn->start_stop_hotkey, 0, ags.size()-1, agn->n_id);
    }
}

void application_graph_save(string base_dir, string name) {
    std::ofstream outfile;

    stringstream s_fullpath;
    s_fullpath << base_dir << "/" << name << ".ags";

    outfile.open(s_fullpath.str().c_str(), std::ios_base::out);

    for (int ag_id = 0; ag_id < ags.size(); ag_id++) {
        outfile << ags[ag_id]->name << std::endl;

        stringstream g_fullpath;
        g_fullpath << base_dir << "/" << name << "_" << ags[ag_id]->name << ".agns";
        std::ofstream g_outfile;
        g_outfile.open(g_fullpath.str().c_str(), std::ios_base::out);

        stringstream ns_fullpath;
        ns_fullpath << base_dir << "/" << name << "_" << ags[ag_id]->name << ".node_settings";
        std::ofstream ns_outfile;
        ns_outfile.open(ns_fullpath.str().c_str(), std::ios_base::out);

        for (int n_id = 0; n_id < ags[ag_id]->nodes.size(); n_id++) {
            struct application_graph_node* agn = ags[ag_id]->nodes[n_id];

            if (agn == nullptr) continue;
            ns_outfile << agn->n_id << std::endl;
            string ns_out_str;
            application_graph_node_settings_externalise(agn, ns_out_str);
            ns_outfile << ns_out_str;

            g_outfile << agn->n_id << std::endl;

            g_outfile << agn->pos_x << std::endl;
            g_outfile << agn->pos_y << std::endl;

            g_outfile << agn->component_type << std::endl;

            string component_str;

            if (agn->externalise != nullptr) {
#ifndef FPTR_EXT_C
#define FPTR_EXT_C
                typedef void (*extcptr)(struct application_graph_node* agn, string &out_string);
#endif
                extcptr f_ptr = (extcptr)agn->externalise;
                f_ptr(agn, component_str);

                g_outfile << component_str;
            }
        }

        stringstream e_fullpath;
        e_fullpath << base_dir << "/" << name << "_" << ags[ag_id]->name << ".edges";
        std::ofstream e_outfile;
        e_outfile.open(e_fullpath.str().c_str(), std::ios_base::out);
        for (int e_id = 0; e_id < ags[ag_id]->edges.size(); e_id++) {
            struct application_graph_edge* edge = ags[ag_id]->edges[e_id];

            if (edge == nullptr) continue;
            e_outfile << edge->from.first->n_id << std::endl;
            e_outfile << edge->from.second << std::endl;
            e_outfile << edge->to.first->n_id << std::endl;
            e_outfile << edge->to.second << std::endl;
        }
    }
}

void application_graph_load(string base_dir, string name) {
    std::ifstream infile;

    stringstream s_fullpath;
    s_fullpath << base_dir << "/" << name << ".ags";

    infile.open(s_fullpath.str().c_str(), std::ios_base::in);
    string i_line;
    int ag_c = 0;
    while (std::getline(infile, i_line)) {
        if (strlen(i_line.c_str()) == 0) return;
        struct application_graph* ag = new application_graph();
        ag->name = i_line;

        stringstream g_fullpath;
        g_fullpath << base_dir << "/" << name << "_" << i_line << ".agns";
        std::ifstream g_infile;
        g_infile.open(g_fullpath.str().c_str(), std::ios_base::in);
        string g_line;
        while (std::getline(g_infile, g_line)) {
            if (strlen(g_line.c_str()) == 0) break;
            struct application_graph_node* agn = new application_graph_node();
            agn->start_stop_hotkey = -1;
            int n_id = stoi(g_line);
            agn->n_id = n_id;
            std::getline(g_infile, g_line);
            int pos_x = stoi(g_line);
            std::getline(g_infile, g_line);
            int pos_y = stoi(g_line);
            std::getline(g_infile, g_line);
            enum application_graph_component_type agct = (enum application_graph_component_type) stoi(g_line);
            switch (agct) {
                case AGCT_GPU_COMPOSER_ELEMENT: {
                    struct gpu_composer_element* gce = new gpu_composer_element();
                    gpu_composer_element_load(gce, g_infile);
                    gpu_composer_element_ui_graph_init(agn, (application_graph_component)gce, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_COMPOSER: {
                    struct gpu_composer* gc = new gpu_composer();
                    gpu_composer_load(gc, g_infile);
                    gpu_composer_ui_graph_init(agn, (application_graph_component)gc, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_DENOISE: {
                    struct gpu_denoise* gd = new gpu_denoise();
                    gpu_denoise_load(gd, g_infile);
                    gpu_denoise_ui_graph_init(agn, (application_graph_component)gd, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_MEMORY_BUFFER: {
                    struct gpu_memory_buffer* gmb = new gpu_memory_buffer();
                    gpu_memory_buffer_load(gmb, g_infile);
                    gpu_memory_buffer_ui_graph_init(agn, (application_graph_component)gmb, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_VIDEO_ALPHA_MERGE: {
                    struct gpu_video_alpha_merge* vam = new gpu_video_alpha_merge();
                    gpu_video_alpha_merge_load(vam, g_infile);
                    gpu_video_alpha_merge_ui_graph_init(agn, (application_graph_component)vam, pos_x, pos_y);
                    break;
                }
                case AGCT_IM_SHOW: {
                    struct im_show* is = new im_show();
                    im_show_load(is, g_infile);
                    im_show_ui_graph_init(agn, (application_graph_component)is, pos_x, pos_y);
                    break;
                }
                case AGCT_MASK_RCNN: {
                    struct mask_rcnn* mrcnn = new mask_rcnn();
                    mask_rcnn_load(mrcnn, g_infile);
                    mask_rcnn_ui_graph_init(agn, (application_graph_component)mrcnn, pos_x, pos_y);
                    break;
                }
                case AGCT_SHARED_MEMORY_BUFFER: {
                    struct shared_memory_buffer* smb = new shared_memory_buffer();
                    shared_memory_buffer_load(smb, g_infile);
                    shared_memory_buffer_ui_graph_init(agn, (application_graph_component)smb, pos_x, pos_y);
                    break;
                }
                case AGCT_VIDEO_SOURCE: {
                    struct video_source* vs = new video_source();
                    video_source_load(vs, g_infile);
                    video_source_ui_graph_init(agn, (application_graph_component)vs, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_MOTION_BLUR: {
                    struct gpu_motion_blur* mb = new gpu_motion_blur();
                    gpu_motion_blur_load(mb, g_infile);
                    gpu_motion_blur_ui_graph_init(agn, (application_graph_component)mb, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_GAUSSIAN_BLUR: {
                    struct gpu_gaussian_blur* gb = new gpu_gaussian_blur();
                    gpu_gaussian_blur_load(gb, g_infile);
                    gpu_gaussian_blur_ui_graph_init(agn, (application_graph_component)gb, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_EDGE_FILTER: {
                    struct gpu_edge_filter* gef = new gpu_edge_filter();
                    gpu_edge_filter_load(gef, g_infile);
                    gpu_edge_filter_ui_graph_init(agn, (application_graph_component)gef, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_PALETTE_FILTER: {
                    struct gpu_palette_filter* gpf = new gpu_palette_filter();
                    gpu_palette_filter_load(gpf, g_infile);
                    gpu_palette_filter_ui_graph_init(agn, (application_graph_component)gpf, pos_x, pos_y);
                    break;
                }
                case AGCT_GPU_AUDIOVISUAL: {
                    struct gpu_audiovisual* gav = new gpu_audiovisual();
                    gpu_audiovisual_load(gav, g_infile);
                    gpu_audiovisual_ui_graph_init(agn, (application_graph_component)gav, pos_x, pos_y);
                    break;
                }
                case AGCT_AUDIO_SOURCE: {
                    struct audio_source* gas = new audio_source();
                    audio_source_load(gas, g_infile);
                    audio_source_ui_graph_init(agn, (application_graph_component)gas, pos_x, pos_y);
                    break;
                }
                case AGCT_MINI_GINE: {
                    struct mini_gine* mg = new mini_gine();
                    mini_gine_load(mg, g_infile);
                    mini_gine_ui_graph_init(agn, (application_graph_component)mg, pos_x, pos_y);
                    break;
                }
            }
            while (ag->nodes.size() < agn->n_id) {
                ag->nodes.push_back(nullptr);
            }
            ag->nodes.push_back(agn);
        }
        ags.push_back(ag);

        stringstream e_fullpath;
        e_fullpath << base_dir << "/" << name << "_" << i_line << ".edges";
        std::ifstream e_infile;
        e_infile.open(e_fullpath.str().c_str(), std::ios_base::in);
        string e_line;
        while (std::getline(e_infile, e_line)) {
            if (strlen(e_line.c_str()) == 0) break;
            int from_nid = stoi(e_line);
            std::getline(e_infile, e_line);
            int from_id = stoi(e_line);
            std::getline(e_infile, e_line);
            int to_nid = stoi(e_line);
            std::getline(e_infile, e_line);
            int to_id = stoi(e_line);
            application_graph_add_edge_intl(ags.size()-1, from_nid, to_nid, from_id, to_id);
        }
        
        stringstream ns_fullpath;
        ns_fullpath << base_dir << "/" << name << "_" << i_line << ".node_settings";
        std::ifstream ns_infile;
        ns_infile.open(ns_fullpath.str().c_str(), std::ios_base::in);
        string ns_line;
        while (std::getline(ns_infile, ns_line)) {
            if (strlen(ns_line.c_str()) == 0) break;
            int n_id = stoi(ns_line);

            struct application_graph_node *agn = ags[ag_c]->nodes[n_id];
            application_graph_node_settings_load(agn, ns_infile);
        }
        ag_c++;
    }
    myApp->drawPane->Refresh();
}